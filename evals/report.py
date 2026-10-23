import json

MAX_CELL = 60

def _cell(text):
    """Flatten a model output into a single Markdown table cell."""
    s = " ".join(str(text).split())
    s = s.replace("|", "\\|")
    if len(s) > MAX_CELL:
        s = s[:MAX_CELL - 1].rstrip() + " ..."
    return s

def _usd(value):
    return "n/a" if value is None else "$%.6f" % value

def _scorer_breakdown(rows):
    """Per-scorer counts as [passed, failed], keyed by scorer name."""
    stats = {}
    for r in rows:
        for name, ok in json.loads(r[3] or "{}").items():
            s = stats.setdefault(name, [0, 0])
            s[0 if ok else 1] += 1
    return stats

def render_markdown(run, rows):
    """Render one run as a Markdown report.

    `run` is a runs-table row and `rows` the matching results-table rows, as
    returned by Ledger.run_by_ref / Ledger.results_for.
    """
    ref, timestamp, provider, model = run[1], run[2], run[3], run[4]
    total, passed, failed, wall, cost = run[5], run[6], run[7], run[8], run[9]
    priced = [r for r in rows if r[7] is not None]
    unpriced = len(rows) - len(priced)
    pt = sum(r[5] or 0 for r in rows)
    ct = sum(r[6] or 0 for r in rows)

    out = ["# Eval run `%s`" % ref, ""]
    out += [
        "| | |",
        "| --- | --- |",
        "| Timestamp | %s |" % timestamp,
        "| Provider | %s |" % provider,
        "| Model | %s |" % model,
        "| Passed | %d/%d (%s) |" % (passed, total, "n/a" if not total else "%.1f%%" % (100.0 * passed / total)),
        "| Failed | %d |" % failed,
        "| Wall time | %.3fs |" % wall,
        "| Cost | %s |" % _usd(cost),
        "",
        "## Cases",
        "",
        "| Case | Status | Scorers | Output | Latency | Tokens | Cost |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        scorers = json.loads(r[3] or "{}")
        detail = ", ".join("%s %s" % (k, "pass" if v else "fail") for k, v in sorted(scorers.items()))
        tokens = "%d/%d" % (r[5] or 0, r[6] or 0)
        out.append("| %s | %s | %s | %s | %.0fms | %s | %s |" % (
            _cell(r[0]), "**PASS**" if r[2] else "**FAIL**", _cell(detail) or "-",
            _cell(r[1]), r[4] or 0.0, tokens, _usd(r[7])))

    stats = _scorer_breakdown(rows)
    out += ["", "## Scorers", "", "| Scorer | Pass | Fail | Pass rate |", "| --- | ---: | ---: | ---: |"]
    if not stats:
        out.append("| - | - | - | - |")
    for name in sorted(stats):
        ok, bad = stats[name]
        rate = "%.1f%%" % (100.0 * ok / (ok + bad))
        out.append("| %s | %d | %d | %s |" % (_cell(name), ok, bad, rate))

    out += [
        "",
        "## Cost",
        "",
        "- Prompt tokens: %d" % pt,
        "- Completion tokens: %d" % ct,
        "- Run total: %s" % _usd(cost),
    ]
    if priced:
        top = max(priced, key=lambda r: r[7])
        out.append("- Most expensive case: `%s` (%s)" % (_cell(top[0]), _usd(top[7])))
    if unpriced:
        out.append("- %d case(s) on models without a price entry, cost reported as n/a" % unpriced)
    out.append("")
    return "\n".join(out)
