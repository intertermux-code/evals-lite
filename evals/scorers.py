import re

def _normalize(s):
    return " ".join(str(s).lower().split())

def exact(output, expected, **kw):
    return _normalize(output) == _normalize(expected)

def contains(output, expected, **kw):
    return _normalize(expected) in _normalize(output)

def fuzzy(output, expected, threshold=0.6, **kw):
    ot = set(_normalize(output).split())
    et = set(_normalize(expected).split())
    if not et:
        return True
    overlap = len(ot & et)
    return (overlap / len(et)) >= threshold

SCORERS = {
    "exact": exact,
    "contains": contains,
    "fuzzy": fuzzy,
}

def run_scorer(name, output, expected, **kw):
    if name not in SCORERS:
        raise ValueError(f"unknown scorer: {name}")
    return bool(SCORERS[name](output, expected, **kw))
