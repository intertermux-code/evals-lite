import argparse
import json
import os
import sys
import time

from evals import __version__
from evals.scorers import run_scorer
from evals.provider import Provider, MockProvider
from evals.ledger import Ledger, cost_for
from evals.yamlparse import parse

DEFAULT_EVAL = "evals.yaml"
DEFAULT_DB = ".evals/evals.db"

def load_evals(path):
    with open(path) as f:
        data = parse(f.read())
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("eval file must contain 'cases' list")
    cases = data["cases"]
    if not isinstance(cases, list):
        raise ValueError("'cases' must be a list")
    for c in cases:
        if not isinstance(c, dict) or "id" not in c or "input" not in c:
            raise ValueError("each case needs 'id' and 'input'")
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

def run_one(case, provider):
    scorers = case.get("scorers")
    if scorers is None:
        s = case.get("scorer", "exact")
        scorers = [s] if isinstance(s, str) else (s or ["exact"])
    if not isinstance(scorers, list):
        scorers = [scorers]
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
    data = load_evals(args.eval)
    provider = MockProvider(model=args.model or "mock") if args.dry_run else Provider("default", "https://api.openai.com", args.model or "gpt-4o-mini", api_key=args.api_key)
    # allow provider config? keep simple
    cases = data["cases"]
    ledger = Ledger(args.db)
    t0 = time.time()
    results = []
    for c in cases:
        r = run_one(c, provider)
        results.append(r)
    wall = time.time() - t0
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    total_cost = None
    costs = [r["cost_usd"] for r in results if r["cost_usd"] is not None]
    if costs:
        total_cost = sum(costs)
    ref = args.ref or time.strftime("%Y%m%dT%H%M%S")
    run_id = ledger.create_run(ref, provider.name, provider.model, len(results), passed, failed, wall, total_cost)
    for r in results:
        ledger.add_result(run_id, r["case_id"], r["output"], r["passed"], r["scorer_results"], r["latency_ms"], r["prompt_tokens"], r["completion_tokens"], r["cost_usd"])
    ledger.close()
    print(f"run {ref}: {passed}/{len(results)} passed")
    return 0

def cmd_list(args):
    ledger = Ledger(args.db)
    rows = ledger.runs()
    ledger.close()
    if args.json:
        out = [{"ref": r[1], "timestamp": r[2], "passed": r[6], "total": r[5], "model": r[4]} for r in rows]
        print(json.dumps(out, indent=2))
        return 0
    print(f"{'ref':<20} {'timestamp':<22} {'pass/total':<12} {'model'}")
    for r in rows:
        print(f"{r[1]:<20} {r[2]:<22} {r[6]}/{r[5]:<10} {r[4]}")
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
    for r in rows:
        print(f"{r[0]}: {'PASS' if r[2] else 'FAIL'}  latency={r[4]:.0f}ms")
        print(f"  output: {r[1]}")
    return 0

def cmd_cost(args):
    ledger = Ledger(args.db)
    run = ledger.run_by_ref(args.ref) if args.ref else ledger.runs()[-1] if ledger.runs() else None
    if not run:
        print("run not found", file=sys.stderr); return 1
    rows = ledger.results_for(run[0])
    ledger.close()
    total = run[9]
    if args.json:
        out = {"ref": run[1], "total_cost_usd": total, "cases": [{"case_id": r[0], "cost_usd": r[7]} for r in rows]}
        print(json.dumps(out, indent=2))
        return 0
    print(f"ref={run[1]} total_cost_usd={total}")
    for r in rows:
        print(f"  {r[0]}: {r[7]}")
    return 0

def cmd_diff(args):
    ledger = Ledger(args.db)
    a = ledger.run_by_ref(args.ref1)
    b = ledger.run_by_ref(args.ref2)
    if not a or not b:
        print("run not found", file=sys.stderr); ledger.close(); return 1
    ra = {r[0]: r for r in ledger.results_for(a[0])}
    rb = {r[0]: r for r in ledger.results_for(b[0])}
    ledger.close()
    regressions = []
    fixed = []
    new = []
    removed = []
    for cid in sorted(set(ra) | set(rb)):
        if cid in ra and cid in rb:
            pa = bool(ra[cid][2]); pb = bool(rb[cid][2])
            if pa and not pb:
                regressions.append(cid)
            elif (not pa) and pb:
                fixed.append(cid)
        elif cid in rb and cid not in ra:
            new.append(cid)
        elif cid in ra and cid not in rb:
            removed.append(cid)
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
    sp.add_argument("--api-key", default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--model", default=None)
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
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_cost)
    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
