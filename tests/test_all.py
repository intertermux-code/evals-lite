import os, tempfile, unittest, subprocess, sys
from evals.scorers import exact, contains, fuzzy, run_scorer
from evals.yamlparse import parse
from evals.ledger import Ledger, cost_for
from evals import cli as evals_cli
from evals.cli import main, run_one
from evals.provider import MockProvider, Provider

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

class TestRunCLIFlags(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, self.cwd)
        main(["init"])

    def _record_provider(self, seen):
        """A provider that remembers its endpoint and answers without a network."""
        class Recorder(Provider):
            def __init__(self, name, base_url, model, api_key=None):
                seen["base_url"] = base_url
                super().__init__(name, base_url, model, api_key)

            def chat(self, prompt):
                return {"output": "hello", "prompt_tokens": 1, "completion_tokens": 1}

        original = evals_cli.Provider
        evals_cli.Provider = Recorder
        self.addCleanup(setattr, evals_cli, "Provider", original)

    def test_concurrency_above_one_is_refused(self):
        rc = main(["run", "smoke", "--dry-run", "--concurrency", "4",
                   "--db", ".evals/evals.db"])
        self.assertEqual(rc, 1)

    def test_concurrency_of_one_is_accepted(self):
        rc = main(["run", "smoke", "--dry-run", "--concurrency", "1",
                   "--db", ".evals/evals.db"])
        self.assertEqual(rc, 0)

    def test_base_url_reaches_the_provider(self):
        seen = {}
        self._record_provider(seen)
        rc = main(["run", "smoke", "--base-url", "http://127.0.0.1:11434/v1",
                   "--db", ".evals/evals.db"])
        self.assertEqual(rc, 0)
        self.assertEqual(seen["base_url"], "http://127.0.0.1:11434/v1")

    def test_default_base_url_is_the_openai_endpoint(self):
        seen = {}
        self._record_provider(seen)
        rc = main(["run", "smoke", "--db", ".evals/evals.db"])
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

if __name__ == "__main__":
    unittest.main()
