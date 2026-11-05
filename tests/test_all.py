import os, re, tempfile, unittest, subprocess, sys, io, json, contextlib, threading
from unittest import mock
from evals.scorers import exact, contains, fuzzy, run_scorer
from evals.scorers import resolve as resolve_scorer, names as scorer_names
from evals.yamlparse import parse
from evals.ledger import Ledger, cost_for
from evals import cli as evals_cli
from evals import __version__
from evals.cli import main, run_one, load_evals, ConfigError
from evals.presets import resolve, names
from evals.provider import MockProvider, Provider
from evals.report import render_markdown

def invoke(argv):
    """Run the CLI and hand back (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()

def record_provider(seen):
    """Swap in a Provider that remembers its config and answers without network.

    Returns a callable that restores the original class.
    """
    class Recorder(Provider):
        def __init__(self, name, base_url, model, api_key=None):
            seen.update(name=name, base_url=base_url, model=model, api_key=api_key)
            super().__init__(name, base_url, model, api_key)

        def chat(self, prompt):
            return {"output": "hello", "prompt_tokens": 1, "completion_tokens": 1}

    original = evals_cli.Provider
    evals_cli.Provider = Recorder
    return lambda: setattr(evals_cli, "Provider", original)

class Sandbox(unittest.TestCase):
    """Base for CLI tests that scaffold a project in a temp directory."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, self.cwd)
        self.db = ".evals/evals.db"
        invoke(["init"])

class TestMultipleScorers(unittest.TestCase):
    def test_all_must_pass(self):
        # mock output is "mock response: <prompt>": contains("mock") passes,
        # exact fails -> case fails (AND semantics)
        case = {"id": "c1", "input": "hi", "expected": "mock", "scorers": ["exact", "contains"]}
        r = run_one(case, MockProvider())
        self.assertFalse(r["passed"])
        self.assertTrue(r["scorer_results"]["contains"])
        self.assertFalse(r["scorer_results"]["exact"])
        case["expected"] = "mock response: hi"
        r = run_one(case, MockProvider())
        self.assertTrue(r["passed"])

class TestScorers(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(exact("Hello World", "hello   world"))
        self.assertFalse(exact("a", "b"))
    def test_contains(self):
        self.assertTrue(contains("abc def", "bc d"))
        self.assertFalse(contains("abc", "xyz"))
    def test_fuzzy_boundary(self):
        # exactly at threshold counts as pass
        self.assertTrue(fuzzy("a b c d e f", "a b c d e f g h i j", threshold=0.6))
        self.assertFalse(fuzzy("a b c d e", "a b c d e f g h i j", threshold=0.6))
    def test_run_scorer_unknown(self):
        with self.assertRaises(ValueError):
            run_scorer("nope", "a", "b")

class TestYaml(unittest.TestCase):
    def test_parse(self):
        t = "cases:\n  - id: c1\n    input: hi\n    expected: hello\n    scorer: contains\n"
        d = parse(t)
        self.assertEqual(d["cases"][0]["id"], "c1")
    def test_malformed(self):
        # lenient parser: garbage in still returns a dict, never raises
        self.assertIsInstance(parse("cases: ["), dict)
        self.assertIsInstance(parse("{{{\n  :::"), dict)

class TestLedger(unittest.TestCase):
    def test_roundtrip(self):
        d = tempfile.mkdtemp()
        db = os.path.join(d, "t.db")
        L = Ledger(db)
        rid = L.create_run("r1", "mock", "mock", 1, 1, 0, 1.0, 0.0)
        L.add_result(rid, "c1", "out", True, {"exact": True}, 5.0, 10, 5, 0.0)
        rows = L.results_for(rid)
        self.assertEqual(rows[0][0], "c1")
        L.close()

    def test_result_counts(self):
        d = tempfile.mkdtemp()
        db = os.path.join(d, "t.db")
        L = Ledger(db)
        r1 = L.create_run("full", "mock", "mock", 2, 2, 0, 1.0, 0.0)
        L.add_result(r1, "c1", "out", True, {"exact": True}, 5.0, 10, 5, 0.0)
        L.add_result(r1, "c2", "out", True, {"exact": True}, 5.0, 10, 5, 0.0)
        r2 = L.create_run("cut", "mock", "mock", 3, 1, 0, 1.0, 0.0)
        L.add_result(r2, "c1", "out", True, {"exact": True}, 5.0, 10, 5, 0.0)
        self.assertEqual(L.result_counts(), {r1: 2, r2: 1})
        L.close()

class TestCost(unittest.TestCase):
    def test_math(self):
        self.assertAlmostEqual(cost_for("mock", 1000, 1000), 0.0)
        self.assertIsNone(cost_for("unknown-model", 10, 10))

class TestRunCLIFlags(Sandbox):
    def test_concurrency_above_one_is_refused(self):
        rc = main(["run", "smoke", "--dry-run", "--concurrency", "4",
                   "--db", self.db])
        self.assertEqual(rc, 1)

    def test_concurrency_of_one_is_accepted(self):
        rc = main(["run", "smoke", "--dry-run", "--concurrency", "1",
                   "--db", self.db])
        self.assertEqual(rc, 0)

    def test_base_url_reaches_the_provider(self):
        seen = {}
        self.addCleanup(record_provider(seen))
        rc = main(["run", "smoke", "--base-url", "http://127.0.0.1:11434/v1",
                   "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertEqual(seen["base_url"], "http://127.0.0.1:11434/v1")

    def test_default_base_url_is_the_openai_endpoint(self):
        seen = {}
        self.addCleanup(record_provider(seen))
        rc = main(["run", "smoke", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertEqual(seen["base_url"], "https://api.openai.com")


class TestDiffCLI(unittest.TestCase):
    def test_e2e(self):
        d = tempfile.mkdtemp()
        os.chdir(d)
        # init
        main(["init"])
        # run smoke dry-run
        rc = main(["run", "smoke", "--dry-run", "--db", ".evals/evals.db"])
        self.assertEqual(rc, 0)
        # same run twice -> clean diff, no regression
        main(["run", "smoke2", "--dry-run", "--db", ".evals/evals.db"])
        rc = main(["diff", "smoke", "smoke2", "--fail-on-regression", "--db", ".evals/evals.db"])
        self.assertEqual(rc, 0)
    def test_regression_exit_code(self):
        # make the greeting case fail in the next run -> diff must exit 1
        with open("evals.yaml") as f:
            text = f.read()
        with open("evals.yaml", "w") as f:
            f.write(text.replace('expected: "hello"', 'expected: "zzznomatch"'))
        main(["run", "smoke3", "--dry-run", "--db", ".evals/evals.db"])
        rc = main(["diff", "smoke", "smoke3", "--fail-on-regression", "--db", ".evals/evals.db"])
        self.assertEqual(rc, 1)
        # without the flag the gate is report-only
        rc = main(["diff", "smoke", "smoke3", "--db", ".evals/evals.db"])
        self.assertEqual(rc, 0)

class TestDiffDuplicateIds(Sandbox):
    def _case(self, cid, expected):
        return '  - id: %s\n    input: "say hi"\n    expected: "%s"\n    scorer: contains\n' % (cid, expected)

    def _write_eval(self, lines):
        with open("evals.yaml", "w") as f:
            f.write("cases:\n" + "".join(lines))

    def test_duplicated_id_regression_is_not_swallowed(self):
        # both occurrences pass in r1; in r2 the FIRST occurrence fails while
        # the last one passes -- keyed-by-id diffs used to see pass->pass
        self._write_eval([self._case("dup", "mock"), self._case("dup", "mock")])
        rc, out, _ = invoke(["run", "r1", "--dry-run", "--eval", "evals.yaml", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("2/2 passed", out)
        self._write_eval([self._case("dup", "zzz"), self._case("dup", "mock")])
        rc, out, _ = invoke(["run", "r2", "--dry-run", "--eval", "evals.yaml", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("1/2 passed", out)
        rc, out, _ = invoke(["diff", "r1", "r2", "--fail-on-regression", "--db", self.db])
        self.assertEqual(rc, 1)
        self.assertIn("REGRESSION: dup#1", out)
        self.assertIn("regressions=1", out)

    def test_occurrences_pair_in_order_and_extras_are_listed(self):
        self._write_eval([self._case("dup", "zzz"), self._case("dup", "mock"),
                          self._case("solo", "mock")])
        invoke(["run", "r1", "--dry-run", "--eval", "evals.yaml", "--db", self.db])
        self._write_eval([self._case("dup", "zzz"), self._case("dup", "zzz"),
                          self._case("dup", "mock")])
        invoke(["run", "r2", "--dry-run", "--eval", "evals.yaml", "--db", self.db])
        rc, out, _ = invoke(["diff", "r1", "r2", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("REGRESSION: dup#2", out)
        self.assertNotIn("REGRESSION: dup#1", out)
        self.assertIn("NEW: dup#3", out)
        self.assertIn("REMOVED: solo", out)
        self.assertIn("regressions=1 fixed=0 new=1 removed=1", out)
        rc, out, _ = invoke(["diff", "r1", "r2", "--json", "--db", self.db])
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertEqual(doc["regressions"], ["dup#2"])
        self.assertEqual(doc["new"], ["dup#3"])
        self.assertEqual(doc["removed"], ["solo"])

    def test_unique_ids_keep_the_plain_labels(self):
        rc, _, _ = invoke(["run", "r1", "--dry-run", "--eval", "evals.yaml", "--db", self.db])
        self.assertEqual(rc, 0)
        with open("evals.yaml") as f:
            text = f.read()
        with open("evals.yaml", "w") as f:
            f.write(text.replace('expected: "hello"', 'expected: "zzznomatch"'))
        invoke(["run", "r2", "--dry-run", "--eval", "evals.yaml", "--db", self.db])
        rc, out, _ = invoke(["diff", "r1", "r2", "--fail-on-regression", "--db", self.db])
        self.assertEqual(rc, 1)
        self.assertIn("REGRESSION: greeting\n", out)
        self.assertNotIn("#", out)


class TestProviderPresets(unittest.TestCase):
    def test_defaults_come_from_the_openai_preset(self):
        cfg = resolve(env={})
        self.assertEqual(cfg.name, "openai")
        self.assertEqual(cfg.base_url, "https://api.openai.com")
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertIsNone(cfg.api_key)

    def test_preset_supplies_endpoint_and_model(self):
        cfg = resolve("anthropic", env={"ANTHROPIC_API_KEY": "sk-ant-test"})
        self.assertEqual(cfg.base_url, "https://api.anthropic.com")
        self.assertEqual(cfg.model, "claude-sonnet-4-5")
        self.assertEqual(cfg.api_key, "sk-ant-test")

    def test_local_preset_needs_no_key(self):
        cfg = resolve("ollama", env={"OPENAI_API_KEY": "sk-other", "ANTHROPIC_API_KEY": "sk-ant"})
        self.assertEqual(cfg.base_url, "http://localhost:11434")
        self.assertIsNone(cfg.api_key)

    def test_flags_win_over_the_preset(self):
        cfg = resolve("anthropic", base_url="http://127.0.0.1:8080/v1",
                      model="claude-from-scratch", api_key="inline", env={})
        self.assertEqual(cfg.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(cfg.model, "claude-from-scratch")
        self.assertEqual(cfg.api_key, "inline")

    def test_unknown_provider_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            resolve("mistral", env={})
        self.assertIn("mistral", str(ctx.exception))
        for name in names():
            self.assertIn(name, str(ctx.exception))


class TestRunProviderFlags(Sandbox):
    def setUp(self):
        super().setUp()
        self.seen = {}
        self.addCleanup(record_provider(self.seen))

    def _provider_row(self, ref):
        ledger = Ledger(self.db)
        row = ledger.run_by_ref(ref)
        ledger.close()
        return row

    def test_provider_selects_endpoint_and_model(self):
        rc, _, _ = invoke(["run", "nightly", "--provider", "anthropic", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertEqual(self.seen["base_url"], "https://api.anthropic.com")
        self.assertEqual(self.seen["model"], "claude-sonnet-4-5")
        row = self._provider_row("nightly")
        self.assertEqual(row[3], "anthropic")
        self.assertEqual(row[4], "claude-sonnet-4-5")

    def test_model_flag_overrides_the_preset_default(self):
        rc, _, _ = invoke(["run", "m", "--provider", "ollama", "--model", "qwen2.5", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertEqual(self.seen["base_url"], "http://localhost:11434")
        self.assertEqual(self.seen["model"], "qwen2.5")

    def test_base_url_overrides_the_preset_endpoint(self):
        rc, _, _ = invoke(["run", "gw", "--provider", "openai", "--base-url", "https://gw.internal",
                           "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertEqual(self.seen["base_url"], "https://gw.internal")

    def test_api_key_env_is_read_per_provider(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env"
        self.addCleanup(os.environ.pop, "ANTHROPIC_API_KEY", None)
        rc, _, _ = invoke(["run", "k", "--provider", "anthropic", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertEqual(self.seen["api_key"], "sk-ant-env")

    def test_explicit_api_key_beats_the_env(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env"
        self.addCleanup(os.environ.pop, "ANTHROPIC_API_KEY", None)
        rc, _, _ = invoke(["run", "k", "--provider", "anthropic", "--api-key", "inline", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertEqual(self.seen["api_key"], "inline")

    def test_unknown_provider_is_refused(self):
        rc, out, err = invoke(["run", "x", "--provider", "nope", "--db", self.db])
        self.assertEqual(rc, 1)
        self.assertIn("unknown provider", err)
        self.assertEqual(out, "")


class TestProviderRequest(Sandbox):
    """A local stand-in server proves the preset config really reaches the wire."""

    def test_endpoint_path_key_env_and_model(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = json.dumps({
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }).encode()
                seen.update(path=self.path, auth=self.headers.get("Authorization"), model=body["model"])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)
        rc, _, err = invoke(["run", "live", "--base-url", "http://127.0.0.1:%d" % server.server_address[1],
                             "--db", self.db])
        self.assertEqual(rc, 0, err)
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer sk-test-key")
        self.assertEqual(seen["model"], "gpt-4o-mini")
        ledger = Ledger(self.db)
        self.assertEqual(ledger.run_by_ref("live")[3], "openai")
        ledger.close()


class TestCostRollups(Sandbox):
    def setUp(self):
        super().setUp()
        ledger = Ledger(self.db)
        r1 = ledger.create_run("nightly", "openai", "gpt-4o-mini", 2, 1, 1, 2.0, 0.00045)
        ledger.add_result(r1, "greeting", "hello", True, {"contains": True}, 120.0, 1000, 500, 0.00015)
        ledger.add_result(r1, "math", "4", False, {"exact": False}, 90.0, 1000, 500, 0.0003)
        r2 = ledger.create_run("local", "ollama", "llama3.1", 1, 1, 0, 1.0, None)
        ledger.add_result(r2, "greeting", "hello", True, {"contains": True}, 300.0, 100, 20, None)
        ledger.close()

    def test_by_model_totals(self):
        rc, out, _ = invoke(["cost", "--by", "model", "--json", "--db", self.db])
        self.assertEqual(rc, 0)
        rows = json.loads(out)
        self.assertEqual([r["model"] for r in rows], ["gpt-4o-mini", "llama3.1"])
        self.assertEqual(rows[0]["runs"], 1)
        self.assertEqual(rows[0]["cases"], 2)
        self.assertEqual(rows[0]["prompt_tokens"], 2000)
        self.assertEqual(rows[0]["completion_tokens"], 1000)
        self.assertAlmostEqual(rows[0]["cost_usd"], 0.00045)
        self.assertIsNone(rows[1]["cost_usd"])
        self.assertEqual(rows[1]["cases"], 1)

    def test_by_run_totals(self):
        rc, out, _ = invoke(["cost", "--by", "run", "--json", "--db", self.db])
        self.assertEqual(rc, 0)
        rows = json.loads(out)
        self.assertEqual([r["ref"] for r in rows], ["nightly", "local"])
        self.assertAlmostEqual(rows[0]["cost_usd"], 0.00045)
        self.assertEqual(rows[0]["passed"], 1)
        self.assertEqual(rows[0]["prompt_tokens"], 2000)
        self.assertIsNone(rows[1]["cost_usd"])
        rc, out, _ = invoke(["cost", "--by", "run", "--db", self.db])
        self.assertEqual(rc, 0)
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "cost by run")
        self.assertIn("nightly", lines[2])
        self.assertIn("0.000450", lines[2])
        self.assertIn("local", lines[3])

    def test_by_model_table_output(self):
        rc, out, _ = invoke(["cost", "--by", "model", "--db", self.db])
        self.assertEqual(rc, 0)
        lines = out.strip().splitlines()
        self.assertEqual(lines[0], "cost by model")
        self.assertIn("completion", lines[1])
        self.assertIn("gpt-4o-mini", lines[2])
        self.assertIn("0.000450", lines[2])
        self.assertIn("llama3.1", lines[3])
        self.assertEqual(lines[3].split()[-1], "-")

    def test_rollups_refuse_a_single_ref(self):
        rc, out, err = invoke(["cost", "nightly", "--by", "model", "--db", self.db])
        self.assertEqual(rc, 1)
        self.assertIn("drop the ref", err)

    def test_single_run_cost_unchanged(self):
        rc, out, _ = invoke(["cost", "nightly", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("ref=nightly total_cost_usd=0.00045", out)

    def test_rollup_on_an_empty_ledger(self):
        rc, out, err = invoke(["cost", "--by", "model", "--db", ".evals/other.db"])
        self.assertEqual(rc, 1)
        self.assertIn("no runs", err)


class TestReportMarkdown(unittest.TestCase):
    def setUp(self):
        self.run = (1, "nightly", "2026-10-01T00:00:00Z", "openai", "gpt-4o", 2, 1, 1, 3.5, 0.00225)
        self.rows = [
            ("alpha", "hello | big\nworld", 1, json.dumps({"contains": True, "exact": False}), 120.0, 1000, 500, 0.0015),
            ("beta", "x" * 200, 0, json.dumps({"contains": False}), 40.0, 500, 250, None),
        ]

    def test_metadata_and_cases(self):
        text = render_markdown(self.run, self.rows)
        self.assertIn("# Eval run `nightly`", text)
        self.assertIn("| Provider | openai |", text)
        self.assertIn("| Model | gpt-4o |", text)
        self.assertIn("| Timestamp | 2026-10-01T00:00:00Z |", text)
        self.assertIn("| Passed | 1/2 (50.0%) |", text)
        self.assertIn("| Case | Status | Scorers | Output | Latency | Tokens | Cost |", text)
        self.assertIn("| alpha | **PASS** |", text)
        self.assertIn("| beta | **FAIL** |", text)
        self.assertIn("120ms", text)

    def test_scorer_breakdown_counts(self):
        text = render_markdown(self.run, self.rows)
        self.assertIn("| Scorer | Pass | Fail | Pass rate |", text)
        self.assertIn("| contains | 1 | 1 | 50.0% |", text)
        self.assertIn("| exact | 0 | 1 | 0.0% |", text)

    def test_cost_summary(self):
        text = render_markdown(self.run, self.rows)
        self.assertIn("- Prompt tokens: 1500", text)
        self.assertIn("- Completion tokens: 750", text)
        self.assertIn("- Run total: $0.002250", text)
        self.assertIn("Most expensive case: `alpha`", text)
        self.assertIn("1 case(s)", text)

    def test_cell_content_is_escaped_and_flattened(self):
        text = render_markdown(self.run, self.rows)
        self.assertIn("hello \\| big world", text)
        self.assertNotIn("hello | big\nworld", text)
        self.assertNotIn("x" * 200, text)
        for line in text.splitlines():
            if line.startswith("| alpha"):
                self.assertLess(len(line), 200)

    def cells(self, text, case_id):
        line = next(l for l in text.splitlines() if l.startswith("| %s " % case_id))
        return [c.strip() for c in line.strip("|").split("|")]

    def test_unpriced_case_cost_shows_na(self):
        text = render_markdown(self.run, self.rows)
        beta = self.cells(text, "beta")
        self.assertEqual(beta[1], "**FAIL**")
        self.assertEqual(beta[2], "contains fail")
        self.assertEqual(beta[4], "40ms")
        self.assertEqual(beta[5], "500/250")
        self.assertEqual(beta[6], "n/a")
        self.assertEqual(self.cells(text, "alpha")[-1], "$0.001500")


class TestReportCLI(Sandbox):
    def setUp(self):
        super().setUp()
        rc, _, _ = invoke(["run", "smoke", "--dry-run", "--db", self.db])
        self.assertEqual(rc, 0)

    def test_markdown_to_stdout(self):
        rc, out, _ = invoke(["report", "smoke", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertTrue(out.startswith("# Eval run `smoke`"))
        self.assertIn("| greeting | **PASS** |", out)
        self.assertIn("| math | **FAIL** |", out)

    def test_defaults_to_the_latest_run(self):
        invoke(["run", "smoke2", "--dry-run", "--db", self.db])
        rc, out, _ = invoke(["report", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("# Eval run `smoke2`", out)

    def test_write_to_file(self):
        rc, out, _ = invoke(["report", "smoke", "--out", "reports/smoke.md", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("reports/smoke.md", out)
        with open("reports/smoke.md") as f:
            text = f.read()
        self.assertIn("# Eval run `smoke`", text)
        self.assertIn("| Case | Status |", text)

    def test_json_format(self):
        rc, out, _ = invoke(["report", "smoke", "--format", "json", "--db", self.db])
        self.assertEqual(rc, 0)
        doc = json.loads(out)
        self.assertEqual(doc["ref"], "smoke")
        self.assertEqual(doc["provider"], "mock")
        self.assertEqual(len(doc["cases"]), 2)
        self.assertTrue(doc["cases"][0]["scorers"]["contains"])

    def test_unknown_run(self):
        rc, out, err = invoke(["report", "nope", "--db", self.db])
        self.assertEqual(rc, 1)
        self.assertIn("run not found", err)


class TestVersion(unittest.TestCase):
    def test_pyproject_matches_package(self):
        """The tag guard in publish.yml reads pyproject, so the two must agree."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "pyproject.toml")) as f:
            declared = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)
        self.assertEqual(declared, __version__)

    def test_cli_version_flag(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stop:
            main(["--version"])
        self.assertEqual(out.getvalue().strip(), __version__)
        self.assertEqual(stop.exception.code, 0)


class FakeUrlopen:
    """Captures the request URL and returns a canned 200 chat-completions response."""

    def __init__(self):
        self.url = None

    def __call__(self, req, timeout=60):
        self.url = req.full_url

        class Response:
            def read(self):
                return b'{"choices": [{"message": {"content": "ok"}}], "usage": {}}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Response()


class TestBaseUrlNormalization(unittest.TestCase):
    def _requested_url(self, base_url):
        catcher = FakeUrlopen()
        with mock.patch("urllib.request.urlopen", side_effect=catcher):
            Provider("openai", base_url, "gpt-4o", api_key="x").chat("hello")
        return catcher.url

    def test_base_url_ending_in_v1_is_used_as_is(self):
        self.assertEqual(self._requested_url("https://api.example.com/v1"),
                         "https://api.example.com/v1/chat/completions")

    def test_base_url_ending_in_v1_with_trailing_slash_is_used_as_is(self):
        self.assertEqual(self._requested_url("https://api.example.com/v1/"),
                         "https://api.example.com/v1/chat/completions")

    def test_base_url_without_v1_keeps_existing_behaviour(self):
        self.assertEqual(self._requested_url("https://api.example.com"),
                         "https://api.example.com/v1/chat/completions")

    def test_trailing_slash_without_v1_is_stripped_then_appended(self):
        self.assertEqual(self._requested_url("https://api.example.com/"),
                         "https://api.example.com/v1/chat/completions")


class TestRequestFailure(Sandbox):
    def _stub_404_server(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return server

    def test_404_exits_cleanly_without_traceback(self):
        server = self._stub_404_server()
        rc, out, err = invoke(["run", "smoke",
                               "--base-url", "http://127.0.0.1:%d" % server.server_address[1],
                               "--db", self.db])
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("failed", err.lower())
        self.assertNotIn("Traceback", err)
        self.assertNotIn("HTTPError", err)

    def test_connection_refused_exits_cleanly_without_traceback(self):
        rc, out, err = invoke(["run", "smoke", "--base-url", "http://127.0.0.1:1", "--db", self.db])
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("failed", err.lower())
        self.assertNotIn("Traceback", err)
        self.assertNotIn("URLError", err)


class StubChatServer:
    """Local OpenAI-compatible endpoint that goes bad partway through a run.

    `state["fail_after"]` is the number of requests still answered with 200;
    every request after that gets a 500, which the provider turns into a
    ProviderError.
    """

    def __init__(self, fail_after=10 ** 6):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        state = {"hits": 0, "fail_after": fail_after}
        self.state = state

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                state["hits"] += 1
                if state["hits"] <= state["fail_after"]:
                    payload = json.dumps({
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    }).encode()
                    self.send_response(200)
                else:
                    payload = b'{"error": {"message": "upstream exploded"}}'
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d" % self.server.server_address[1]

    def shutdown(self):
        self.server.shutdown()

FOUR_CASES = "cases:\n" + "".join(
    '  - id: %s\n    input: "say hi"\n    expected: "hello"\n    scorer: contains\n' % cid
    for cid in ("one", "two", "three", "four"))

class TestPartialRun(Sandbox):
    """A provider error halfway through a long run must not lose the earlier cases."""

    def setUp(self):
        super().setUp()
        with open("evals.yaml", "w") as f:
            f.write(FOUR_CASES)
        self.stub = StubChatServer()
        self.addCleanup(self.stub.shutdown)

    def _run(self, ref, fail_after=2):
        # fail_after counts the requests of this run, not of the server's life
        self.stub.state.update(hits=0, fail_after=fail_after)
        return invoke(["run", ref, "--base-url", self.stub.base_url, "--db", self.db])

    def _ledger_row(self, ref):
        ledger = Ledger(self.db)
        row = ledger.run_by_ref(ref)
        results = ledger.results_for(row[0]) if row else []
        ledger.close()
        return row, results

    def test_results_before_the_error_stay_recoverable(self):
        rc, out, err = self._run("midrun")
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("failed", err.lower())
        self.assertNotIn("Traceback", err)
        row, results = self._ledger_row("midrun")
        self.assertEqual(row[5], 4)  # the planned total, not the recorded one
        self.assertEqual(row[6], 2)
        self.assertEqual(len(results), 2)
        self.assertEqual([r[0] for r in results], ["one", "two"])
        self.assertTrue(all(r[2] for r in results))

    def test_aborted_run_is_flagged_incomplete(self):
        self._run("midrun")
        rc, out, _ = invoke(["list", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("2/4", out)
        self.assertIn("incomplete", out)
        rc, out, _ = invoke(["show", "midrun", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("incomplete: 2 of 4 cases recorded", out)
        self.assertIn("one: PASS", out)
        self.assertIn("two: PASS", out)
        self.assertNotIn("three", out)
        rc, out, _ = invoke(["report", "midrun", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("| Status | incomplete: 2 of 4 cases recorded |", out)

    def test_json_list_flags_the_partial_run_only(self):
        self._run("midrun")
        self._run("complete", fail_after=10 ** 6)
        rc, out, _ = invoke(["list", "--json", "--db", self.db])
        self.assertEqual(rc, 0)
        rows = json.loads(out)
        self.assertEqual([r["ref"] for r in rows], ["midrun", "complete"])
        self.assertTrue(rows[0]["partial"])
        self.assertNotIn("partial", rows[1])

    def test_diff_of_a_partial_run_reports_the_missing_cases(self):
        self._run("full", fail_after=10 ** 6)
        self._run("cut")
        rc, out, _ = invoke(["diff", "full", "cut", "--fail-on-regression", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertIn("REMOVED: three", out)
        self.assertIn("REMOVED: four", out)
        self.assertNotIn("REGRESSION", out)

    def test_failure_on_the_first_case_records_nothing(self):
        rc, out, err = self._run("midrun", fail_after=0)
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertNotIn("partial run", err)
        row, results = self._ledger_row("midrun")
        self.assertIsNone(row)
        self.assertEqual(results, [])

    def test_successful_runs_keep_their_shape(self):
        rc, out, _ = self._run("clean", fail_after=10 ** 6)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "run clean: 4/4 passed\n")
        row, results = self._ledger_row("clean")
        self.assertEqual((row[5], row[6], row[7]), (4, 4, 0))
        self.assertEqual(len(results), 4)
        rc, list_out, _ = invoke(["list", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertNotIn("incomplete", list_out)
        rc, show_out, _ = invoke(["show", "clean", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertNotIn("incomplete", show_out)
        rc, report_out, _ = invoke(["report", "clean", "--db", self.db])
        self.assertEqual(rc, 0)
        self.assertNotIn("incomplete", report_out)


class TestScorerNameMatching(unittest.TestCase):
    """Scorer names are matched case-insensitively; anything else is a config error."""

    def test_every_spelling_resolves_to_one_scorer(self):
        for spelling in ("exact", "Exact", "EXACT", "eXaCt"):
            self.assertEqual(resolve_scorer(spelling), "exact")
        self.assertEqual(resolve_scorer("Contains"), "contains")
        self.assertEqual(resolve_scorer("FUZZY"), "fuzzy")

    def test_mis_cased_names_score_like_the_canonical_one(self):
        for spelling in ("Exact", "EXACT"):
            self.assertTrue(run_scorer(spelling, "Hello   World", "hello world"))
            self.assertFalse(run_scorer(spelling, "hello", "world"))
        self.assertTrue(run_scorer("CONTAINS", "abc def", "BC D"))
        self.assertTrue(run_scorer("Fuzzy", "a b c d e f", "a b c d e f g h i j", threshold=0.6))

    def test_canonical_names_keep_their_own_semantics(self):
        # widening name matching must not have changed what a scorer computes
        samples = [("Hello  World", "hello world"), ("abc", "xyz"), ("mock response: hi", "hi")]
        for name, fn in (("exact", exact), ("contains", contains), ("fuzzy", fuzzy)):
            for output, expected in samples:
                self.assertEqual(run_scorer(name, output, expected), bool(fn(output, expected)))

    def test_unknown_name_still_raises_and_lists_the_valid_ones(self):
        for spelling in ("nope", "Nope", "Regex", "e x a c t"):
            with self.assertRaises(ValueError) as ctx:
                run_scorer(spelling, "a", "a")
            self.assertIn(spelling, str(ctx.exception))
            for name in scorer_names():
                self.assertIn(name, str(ctx.exception))

    def test_a_name_that_is_not_a_string_is_unknown(self):
        with self.assertRaises(ValueError):
            run_scorer(3, "a", "a")

TYPO_EVAL = (
    "cases:\n"
    '  - id: greeting\n    input: "Say hello in one word."\n'
    '    expected: "hello"\n    scorer: contains\n'
    '  - id: math\n    input: "What is 2+2?"\n'
    '    expected: "4"\n    scorer: regex\n'
)

MIS_CASED_EVAL = (
    "cases:\n"
    '  - id: greeting\n    input: "Say hello in one word."\n'
    '    expected: "mock response: Say hello in one word."\n    scorer: Exact\n'
)

MIXED_SCORERS_EVAL = (
    "cases:\n"
    '  - id: both\n    input: "hi"\n    expected: "mock"\n'
    "    scorers:\n      - Contains\n      - FUZZY\n"
)

class TestScorerConfigError(Sandbox):
    """A wrong scorer name is a config error: reported, exit 2, never a recorded False."""

    def _eval(self, text):
        with open("evals.yaml", "w") as f:
            f.write(text)

    def _recorded(self, ref):
        ledger = Ledger(self.db)
        row = ledger.run_by_ref(ref)
        results = ledger.results_for(row[0]) if row else []
        ledger.close()
        return row, results

    def test_unknown_scorer_is_reported_and_exits_2(self):
        self._eval(TYPO_EVAL)
        rc, out, err = invoke(["run", "typo", "--dry-run", "--db", self.db])
        self.assertEqual(rc, 2)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn("math", err)
        self.assertIn("regex", err)
        for name in scorer_names():
            self.assertIn(name, err)

    def test_no_result_is_recorded_for_a_bad_scorer_name(self):
        self._eval(TYPO_EVAL)
        self.assertEqual(invoke(["run", "typo", "--dry-run", "--db", self.db])[0], 2)
        row, results = self._recorded("typo")
        self.assertIsNone(row)
        self.assertEqual(results, [])

    def test_the_typo_is_caught_before_the_provider_is_asked(self):
        self._eval(TYPO_EVAL)
        asked = []

        class Counting(Provider):
            def __init__(self, *args, **kwargs):
                pass

            def chat(self, prompt):
                asked.append(prompt)
                return {"output": "hello", "prompt_tokens": 1, "completion_tokens": 1}

        original = evals_cli.Provider
        evals_cli.Provider = Counting
        self.addCleanup(lambda: setattr(evals_cli, "Provider", original))
        rc, out, err = invoke(["run", "typo", "--db", self.db])
        self.assertEqual(rc, 2)
        self.assertEqual(asked, [])

    def test_run_one_refuses_an_unknown_name_instead_of_failing_the_case(self):
        case = {"id": "c1", "input": "hi", "expected": "mock", "scorer": "regex"}
        with self.assertRaises(ConfigError):
            run_one(case, MockProvider())

    def test_mis_cased_scorer_is_scored_and_recorded(self):
        self._eval(MIS_CASED_EVAL)
        rc, out, err = invoke(["run", "cased", "--dry-run", "--db", self.db])
        self.assertEqual(rc, 0, err)
        self.assertEqual(out, "run cased: 1/1 passed\n")
        row, results = self._recorded("cased")
        self.assertTrue(results[0][2])
        self.assertEqual(json.loads(results[0][3]), {"exact": True})

    def test_mis_cased_names_in_a_scorers_list_are_scored(self):
        self._eval(MIXED_SCORERS_EVAL)
        rc, out, err = invoke(["run", "mix", "--dry-run", "--db", self.db])
        self.assertEqual(rc, 0, err)
        self.assertEqual(out, "run mix: 1/1 passed\n")
        row, results = self._recorded("mix")
        self.assertEqual(json.loads(results[0][3]), {"contains": True, "fuzzy": True})

    def test_the_init_sample_still_runs_as_it_did(self):
        rc, out, err = invoke(["run", "smoke", "--dry-run", "--db", self.db])
        self.assertEqual(rc, 0, err)
        self.assertEqual(out, "run smoke: 1/2 passed\n")
        row, results = self._recorded("smoke")
        self.assertEqual((row[5], row[6], row[7]), (2, 1, 1))
        self.assertEqual([json.loads(r[3]) for r in results],
                         [{"contains": True}, {"exact": False}])

BAD_EVAL_FILES = [
    ("top level is a list", "- id: c1\n  input: hi\n", "must contain 'cases'"),
    ("cases is not a list", "cases: 3\n", "'cases' must be a list"),
    ("no cases key", "name: smoke\n", "must contain 'cases'"),
    ("a case is not a mapping", "cases:\n  - hello\n", "case 1 must be a mapping"),
    ("a case has no input", "cases:\n  - id: c1\n", "case 1 needs 'input'"),
    ("a case has neither key", "cases:\n  - expected: hi\n", "needs 'id' and 'input'"),
]

class TestEvalFileErrors(Sandbox):
    """Ordinary eval-file mistakes are reported with the file named, never traced back."""

    def _write(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    def _assert_clean_error(self, rc, out, err, path, needle):
        self.assertEqual(rc, 2)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(path, err)
        self.assertIn(needle, err)

    def test_each_mistake_class_is_reported_and_exits_2(self):
        for index, (label, content, needle) in enumerate(BAD_EVAL_FILES):
            path = "bad_%d.yaml" % index
            self._write(path, content)
            with self.subTest(label):
                rc, out, err = invoke(["run", "bad%d" % index, "--eval", path,
                                       "--dry-run", "--db", self.db])
                self._assert_clean_error(rc, out, err, path, needle)

    def test_a_missing_file_is_reported_and_exits_2(self):
        rc, out, err = invoke(["run", "gone", "--eval", "absent.yaml",
                               "--dry-run", "--db", self.db])
        self._assert_clean_error(rc, out, err, "absent.yaml", "No such file")

    def test_a_directory_is_reported_and_exits_2(self):
        os.makedirs("adir")
        rc, out, err = invoke(["run", "adir", "--eval", "adir",
                               "--dry-run", "--db", self.db])
        self.assertEqual(rc, 2)
        self.assertNotIn("Traceback", err)
        self.assertIn("adir", err)

    def test_a_file_that_breaks_the_parser_is_reported_and_exits_2(self):
        # the mini parser gives up on runaway nesting; that is a mistake to report
        path = "deep.yaml"
        self._write(path, "a:\n" + "".join(" " * (2 * (i + 1)) + "k%d:\n" % i
                                           for i in range(4000)))
        rc, out, err = invoke(["run", "deep", "--eval", path, "--dry-run", "--db", self.db])
        self._assert_clean_error(rc, out, err, path, "malformed YAML")

    def test_no_run_is_recorded_for_a_misshapen_file(self):
        self._write("bad.yaml", "cases: 3\n")
        invoke(["run", "bad", "--eval", "bad.yaml", "--dry-run", "--db", self.db])
        ledger = Ledger(self.db)
        self.assertIsNone(ledger.run_by_ref("bad"))
        ledger.close()

    def test_the_default_eval_path_still_works(self):
        rc, out, err = invoke(["run", "smoke", "--dry-run", "--db", self.db])
        self.assertEqual(rc, 0, err)
        self.assertEqual(out, "run smoke: 1/2 passed\n")

class TestLoadEvals(unittest.TestCase):
    def test_a_misshapen_file_raises_a_config_error_naming_the_file(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "evals.yaml")
        with open(path, "w") as f:
            f.write("cases:\n  - id: c1\n")
        with self.assertRaises(ConfigError) as ctx:
            load_evals(path)
        self.assertIn(path, str(ctx.exception))

    def test_a_config_error_is_still_a_value_error(self):
        # callers that caught ValueError before keep catching the same mistakes
        path = os.path.join(tempfile.mkdtemp(), "absent.yaml")
        with self.assertRaises(ValueError):
            load_evals(path)

    def test_a_valid_file_loads_unchanged(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "evals.yaml")
        with open(path, "w") as f:
            f.write(FOUR_CASES)
        data = load_evals(path)
        self.assertEqual([c["id"] for c in data["cases"]], ["one", "two", "three", "four"])
        self.assertEqual([c["scorer"] for c in data["cases"]], ["contains"] * 4)


if __name__ == "__main__":
    unittest.main()
