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

def names():
    return sorted(SCORERS)

def resolve(name):
    """Canonical scorer key. Names are matched case-insensitively, so a
    capitalized config entry means the same scorer as the lowercase one."""
    key = name.lower() if isinstance(name, str) else name
    if key not in SCORERS:
        raise ValueError("unknown scorer '%s' (valid scorers: %s)" % (name, ", ".join(names())))
    return key

def run_scorer(name, output, expected, **kw):
    return bool(SCORERS[resolve(name)](output, expected, **kw))
