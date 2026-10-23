import os, tempfile, unittest, subprocess, sys, io, json, contextlib, threading
from evals.scorers import exact, contains, fuzzy, run_scorer
from evals.yamlparse import parse
from evals.ledger import Ledger, cost_for
from evals import cli as evals_cli
from evals.cli import main, run_one
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


if __name__ == "__main__":
    unittest.main()
