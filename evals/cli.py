import argparse
import json
import os
import sys
import time

from evals import __version__
from evals.scorers import run_scorer, resolve as resolve_scorer, names as scorer_names
from evals.provider import Provider, MockProvider, ProviderError
from evals.ledger import Ledger, cost_for
from evals.presets import resolve as resolve_provider
from evals.report import render_markdown
from evals.yamlparse import parse

DEFAULT_EVAL = "evals.yaml"
DEFAULT_DB = ".evals/evals.db"

class ConfigError(ValueError):
    """An eval set or scorer name the user can fix; reported, never traced back."""

def load_evals(path):
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        raise ConfigError("%s: %s" % (path, e.strerror or e))
    try:
        data = parse(text)
    except Exception as e:
        raise ConfigError("%s: malformed YAML (%s)" % (path, e))
    if not isinstance(data, dict) or "cases" not in data:
        raise ConfigError("%s: eval file must contain 'cases' list" % path)
    cases = data["cases"]
    if not isinstance(cases, list):
        raise ConfigError("%s: 'cases' must be a list" % path)
    for n, c in enumerate(cases, 1):
        if not isinstance(c, dict):
            raise ConfigError("%s: case %d must be a mapping" % (path, n))
        missing = [k for k in ("id", "input") if k not in c]
        if missing:
            raise ConfigError("%s: case %d needs %s" % (
                path, n, " and ".join("'%s'" % k for k in missing)))
    return data

def cmd_init(args):
    if not os.path.exists("evals.yaml"):
        with open("evals.yaml", "w") as f:
            f.write("""name: smoke
cases:
  - id: greeting
    input: "Say hello in one word."
    expected: "hello"
    scorer: contains
  - id: math
    input: "What is 2+2?"
    expected: "4"
    scorer: exact
""")
    os.makedirs(".evals", exist_ok=True)
    if os.path.exists(".gitignore"):
        with open(".gitignore") as f:
            content = f.read()
        if ".evals/" not in content:
            with open(".gitignore", "a") as f:
                f.write("\n.evals/\n")
    else:
        with open(".gitignore", "w") as f:
            f.write("__pycache__/\n*.pyc\n.evals/\nvenv/\n*.egg-info/\n")
    print("initialized")
    return 0

def scorer_list(case):
    """The scorer entries a case asks for, exactly as written in the eval set."""
    scorers = case.get("scorers")
    if scorers is None:
        s = case.get("scorer", "exact")
        scorers = [s] if isinstance(s, str) else (s or ["exact"])
    if not isinstance(scorers, list):
        scorers = [scorers]
    return scorers

def resolve_scorers(case):
    """Canonical scorer names for one case. A name that matches nothing is a
    config error, so it cannot be mistaken for a model that failed."""
    resolved = []
    for sc in scorer_list(case):
        try:
            resolved.append(resolve_scorer(sc))
        except ValueError:
            raise ConfigError("case '%s': unknown scorer '%s' (valid scorers: %s)" % (
                case.get("id", "?"), sc, ", ".join(scorer_names())))
    return resolved

def run_one(case, provider):
    scorers = resolve_scorers(case)
    expected = case.get("expected", "")
    t0 = time.time()
    res = provider.chat(case["input"])
    latency = (time.time() - t0) * 1000
    output = res["output"]
    sr = {}
    for sc in scorers:
        try:
            sr[sc] = bool(run_scorer(sc, output, expected))
        except Exception as e:
            sr[sc] = False
    passed = all(sr.values()) if sr else True
    c = cost_for(provider.model, res.get("prompt_tokens",0), res.get("completion_tokens",0))
    return {
        "case_id": case["id"],
        "output": output,
        "passed": passed,
        "scorer_results": sr,
        "latency_ms": latency,
        "prompt_tokens": res.get("prompt_tokens",0),
        "completion_tokens": res.get("completion_tokens",0),
        "cost_usd": c,
    }

def cmd_run(args):
    if args.concurrency != 1:
        print("--concurrency is not supported: runs are sequential", file=sys.stderr)
        return 1
    try:
        data = load_evals(args.eval)
        cases = data["cases"]
        # Check every scorer name before the first request: a typo in the eval
        # set should neither spend tokens nor land in the ledger as a failure.
        for c in cases:
            resolve_scorers(c)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    try:
        cfg = resolve_provider(args.provider, args.base_url, args.model, args.api_key)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    provider = MockProvider(model=args.model or "mock") if args.dry_run else Provider(cfg.name, cfg.base_url, cfg.model, api_key=cfg.api_key)
    ledger = Ledger(args.db)
    t0 = time.time()
    results = []
    stopped = None
    for c in cases:
        try:
            r = run_one(c, provider)
        except ProviderError as e:
            stopped = e
            break
        results.append(r)
    wall = time.time() - t0
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    total_cost = None
    costs = [r["cost_usd"] for r in results if r["cost_usd"] is not None]
    if costs:
        total_cost = sum(costs)
    ref = args.ref or time.strftime("%Y%m%dT%H%M%S")
    if stopped is None or results:
        # An aborted run records the planned case count as its total, so the
        # ledger row has fewer results than it promises and reads as incomplete.
        total = len(results) if stopped is None else len(cases)
        run_id = ledger.create_run(ref, provider.name, provider.model, total, passed, failed, wall, total_cost)
        for r in results:
            ledger.add_result(run_id, r["case_id"], r["output"], r["passed"], r["scorer_results"], r["latency_ms"], r["prompt_tokens"], r["completion_tokens"], r["cost_usd"])
    ledger.close()
    if stopped is not None:
        print(stopped, file=sys.stderr)
        if results:
            print(f"partial run {ref}: {len(results)} of {len(cases)} cases recorded", file=sys.stderr)
        return 1
    print(f"run {ref}: {passed}/{len(results)} passed")
    return 0

def cmd_list(args):
    ledger = Ledger(args.db)
    rows = ledger.runs()
    counts = ledger.result_counts()
    ledger.close()
    if args.json:
        out = [{"ref": r[1], "timestamp": r[2], "passed": r[6], "total": r[5], "model": r[4]} for r in rows]
        for item, r in zip(out, rows):
            if counts.get(r[0], 0) < r[5]:
                item["partial"] = True
        print(json.dumps(out, indent=2))
        return 0
    print(f"{'ref':<20} {'timestamp':<22} {'pass/total':<12} {'model'}")
    for r in rows:
        partial = "  incomplete" if counts.get(r[0], 0) < r[5] else ""
        print(f"{r[1]:<20} {r[2]:<22} {r[6]}/{r[5]:<10} {r[4]}{partial}")
    return 0

def cmd_show(args):
    ledger = Ledger(args.db)
    run = ledger.run_by_ref(args.ref)
    if not run:
        print("run not found", file=sys.stderr); return 1
    rows = ledger.results_for(run[0])
    ledger.close()
    if args.json:
        out = [{"case_id": r[0], "output": r[1], "passed": bool(r[2]), "scorers": json.loads(r[3]), "latency_ms": r[4], "prompt_tokens": r[5], "completion_tokens": r[6], "cost_usd": r[7]} for r in rows]
        print(json.dumps(out, indent=2))
        return 0
    print(f"ref={args.ref} model={run[4]}")
    if len(rows) < run[5]:
        print(f"incomplete: {len(rows)} of {run[5]} cases recorded before the run stopped")
    for r in rows:
        print(f"{r[0]}: {'PASS' if r[2] else 'FAIL'}  latency={r[4]:.0f}ms")
        print(f"  output: {r[1]}")
    return 0

def _money(value):
    return "-" if value is None else "%.6f" % value

def cmd_cost_by_model(args, ledger):
    rows = ledger.cost_by_model()
    if args.json:
        print(json.dumps([{"model": r[0], "runs": r[1], "cases": r[2], "prompt_tokens": r[3],
                           "completion_tokens": r[4], "cost_usd": r[5]} for r in rows], indent=2))
        return 0
    if not rows:
        print("no runs recorded", file=sys.stderr)
        return 1
    print("cost by model")
    print(f"{'model':<20} {'runs':>4} {'cases':>5} {'prompt':>7} {'completion':>10} {'cost_usd':>12}")
    for r in rows:
        print(f"{str(r[0]):<20} {r[1]:>4} {r[2]:>5} {r[3]:>7} {r[4]:>10} {_money(r[5]):>12}")
    return 0

def cmd_cost_by_run(args, ledger):
    rows = ledger.cost_by_run()
    if args.json:
        print(json.dumps([{"ref": r[0], "timestamp": r[1], "model": r[2], "total": r[3], "passed": r[4],
                           "prompt_tokens": r[5], "completion_tokens": r[6], "cost_usd": r[7]} for r in rows], indent=2))
        return 0
    if not rows:
        print("no runs recorded", file=sys.stderr)
        return 1
    print("cost by run")
    print(f"{'ref':<20} {'timestamp':<22} {'model':<16} {'pass/total':<11} {'cost_usd':>12}")
    for r in rows:
        print(f"{r[0]:<20} {r[1]:<22} {str(r[2]):<16} {r[4]}/{r[3]:<9} {_money(r[7]):>12}")
    return 0

def cmd_cost(args):
    if args.by and args.ref:
        print("cost --by aggregates over every run; drop the ref argument", file=sys.stderr)
        return 1
    ledger = Ledger(args.db)
    try:
        if args.by == "model":
            return cmd_cost_by_model(args, ledger)
        if args.by == "run":
            return cmd_cost_by_run(args, ledger)
        run = ledger.run_by_ref(args.ref) if args.ref else (ledger.runs()[-1] if ledger.runs() else None)
        if not run:
            print("run not found", file=sys.stderr); return 1
        rows = ledger.results_for(run[0])
        total = run[9]
        if args.json:
            out = {"ref": run[1], "total_cost_usd": total, "cases": [{"case_id": r[0], "cost_usd": r[7]} for r in rows]}
            print(json.dumps(out, indent=2))
            return 0
        print(f"ref={run[1]} total_cost_usd={total}")
        for r in rows:
            print(f"  {r[0]}: {r[7]}")
        return 0
    finally:
        ledger.close()

def cmd_report(args):
    ledger = Ledger(args.db)
    run = ledger.run_by_ref(args.ref) if args.ref else (ledger.runs()[-1] if ledger.runs() else None)
    if not run:
        print("run not found", file=sys.stderr); ledger.close(); return 1
    rows = ledger.results_for(run[0])
    ledger.close()
    if args.format == "json":
        text = json.dumps({
            "ref": run[1], "timestamp": run[2], "provider": run[3], "model": run[4],
            "total": run[5], "passed": run[6], "failed": run[7], "wall_time": run[8], "cost_usd": run[9],
            "cases": [{"case_id": r[0], "output": r[1], "passed": bool(r[2]), "scorers": json.loads(r[3]),
                       "latency_ms": r[4], "prompt_tokens": r[5], "completion_tokens": r[6], "cost_usd": r[7]} for r in rows],
        }, indent=2)
    else:
        text = render_markdown(run, rows)
    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"report written to {args.out}")
    else:
        print(text)
    return 0

def _occurrences(rows):
    # keyed lists keep insertion order; a duplicated id must not lose results
    groups = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r)
    return groups

def _occurrence_label(cid, n, count):
    return f"{cid}#{n}" if count > 1 else cid

def cmd_diff(args):
    ledger = Ledger(args.db)
    a = ledger.run_by_ref(args.ref1)
    b = ledger.run_by_ref(args.ref2)
    if not a or not b:
        print("run not found", file=sys.stderr); ledger.close(); return 1
    ga = _occurrences(ledger.results_for(a[0]))
    gb = _occurrences(ledger.results_for(b[0]))
    ledger.close()
    regressions = []
    fixed = []
    new = []
    removed = []
    for cid in sorted(set(ga) | set(gb)):
        na, nb = ga.get(cid, []), gb.get(cid, [])
        shared = min(len(na), len(nb))
        count = max(len(na), len(nb))
        for i in range(shared):
            label = _occurrence_label(cid, i + 1, count)
            pa = bool(na[i][2]); pb = bool(nb[i][2])
            if pa and not pb:
                regressions.append(label)
            elif (not pa) and pb:
                fixed.append(label)
        for i in range(shared, len(nb)):
            new.append(_occurrence_label(cid, i + 1, count))
        for i in range(shared, len(na)):
            removed.append(_occurrence_label(cid, i + 1, count))
    if args.json:
        print(json.dumps({"regressions": regressions, "fixed": fixed, "new": new, "removed": removed}, indent=2))
    else:
        print(f"Comparing {args.ref1} -> {args.ref2}")
        for cid in regressions:
            print(f"REGRESSION: {cid}")
        for cid in fixed:
            print(f"FIXED: {cid}")
        for cid in new:
            print(f"NEW: {cid}")
        for cid in removed:
            print(f"REMOVED: {cid}")
        print(f"summary: regressions={len(regressions)} fixed={len(fixed)} new={len(new)} removed={len(removed)}")
    if args.fail_on_regression and regressions:
        return 1
    return 0

def main(argv=None):
    p = argparse.ArgumentParser(prog="evals")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init")
    sp.set_defaults(func=cmd_init)
    sp = sub.add_parser("run")
    sp.add_argument("ref", nargs="?")
    sp.add_argument("--eval", default=DEFAULT_EVAL)
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--api-key", default=None,
                    help="overrides the API key read from the provider's env var")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--model", default=None)
    sp.add_argument("--provider", default=None,
                    help="named provider preset (default: openai)")
    sp.add_argument("--base-url", default=None,
                    help="override the provider preset endpoint")
    sp.add_argument("--concurrency", type=int, default=1)
    sp.set_defaults(func=cmd_run)
    sp = sub.add_parser("list")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_list)
    sp = sub.add_parser("diff")
    sp.add_argument("ref1")
    sp.add_argument("ref2")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--fail-on-regression", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_diff)
    sp = sub.add_parser("show")
    sp.add_argument("ref")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_show)
    sp = sub.add_parser("cost")
    sp.add_argument("ref", nargs="?")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--by", choices=("model", "run"), default=None,
                    help="roll up cost over every run instead of a single run")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_cost)
    sp = sub.add_parser("report")
    sp.add_argument("ref", nargs="?")
    sp.add_argument("--db", default=DEFAULT_DB)
    sp.add_argument("--format", choices=("markdown", "json"), default="markdown")
    sp.add_argument("--out", default=None, help="write to a file instead of stdout")
    sp.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
