"""Minimal YAML parser for eval-set files (stdlib only)."""

def parse(text):
    lines = text.splitlines()
    # Remove comments and blank lines but track indentation
    items = []
    # We handle: top-level key: value or key: (then list)
    # Simpler: use a tiny recursive descent for our schema.
    # Strategy: parse lines into nested dict/list via indentation.
    root = {}
    stack = [(0, root, None)]  # (indent, container, key_for_list)
    # We'll process with index
    i = 0
    # Collect non-empty, non-comment lines
    cleaned = []
    for ln in lines:
        if ln.strip() == "" or ln.strip().startswith("#"):
            continue
        cleaned.append(ln)
    # parse
    def parse_block(lines, idx, base_indent):
        # returns (obj, new_idx)
        # determine if list or dict
        if idx >= len(lines):
            return None, idx
        first = lines[idx]
        indent = len(first) - len(first.lstrip())
        if first.lstrip().startswith("- "):
            # list
            lst = []
            while idx < len(lines):
                cur = lines[idx]
                ci = len(cur) - len(cur.lstrip())
                if ci < base_indent:
                    break
                if ci > base_indent:
                    # nested handled by item parse
                    break
                s = cur.lstrip()
                if not s.startswith("- "):
                    break
                # parse item: rest after "- "
                rest = s[2:].strip()
                if rest == "":
                    # nested block follows
                    val, idx = parse_block(lines, idx+1, base_indent+2)
                    lst.append(val)
                elif ":" in rest and not rest.startswith('"') and not rest.startswith("'"):
                    # inline dict start? Actually "- id: x"
                    # Build dict by consuming indented lines
                    item = {}
                    # first key
                    k, v = rest.split(":", 1)
                    k = k.strip(); v = v.strip()
                    if v == "":
                        val, idx2 = parse_block(lines, idx+1, ci+2)
                        item[k] = val
                        idx = idx2
                    else:
                        item[k] = _scalar(v)
                        idx += 1
                    # consume subsequent keys at indent ci+2? Actually keys at same as first? 
                    # In YAML list item, sibling keys are indented to align with first key after "- "
                    key_indent = ci + 2
                    while idx < len(lines):
                        cur2 = lines[idx]
                        ci2 = len(cur2) - len(cur2.lstrip())
                        if ci2 < key_indent:
                            break
                        if ci2 > key_indent:
                            break
                        s2 = cur2.strip()
                        if s2.startswith("- "):
                            break
                        if ":" not in s2:
                            break
                        k2, v2 = s2.split(":", 1)
                        k2 = k2.strip(); v2 = v2.strip()
                        if v2 == "":
                            val, idx2 = parse_block(lines, idx+1, ci2+2)
                            item[k2] = val
                            idx = idx2
                        else:
                            item[k2] = _scalar(v2)
                            idx += 1
                    lst.append(item)
                else:
                    lst.append(_scalar(rest))
                    idx += 1
            return lst, idx
        else:
            # dict
            d = {}
            while idx < len(lines):
                cur = lines[idx]
                ci = len(cur) - len(cur.lstrip())
                if ci < base_indent:
                    break
                if ci > base_indent:
                    break
                s = cur.strip()
                if s.startswith("- "):
                    break
                if ":" not in s:
                    idx += 1
                    continue
                k, v = s.split(":", 1)
                k = k.strip(); v = v.strip()
                if v == "":
                    val, idx2 = parse_block(lines, idx+1, ci+2)
                    d[k] = val
                    idx = idx2
                else:
                    d[k] = _scalar(v)
                    idx += 1
            return d, idx
    obj, _ = parse_block(cleaned, 0, 0)
    return obj if obj is not None else {}

def _scalar(v):
    v = v.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.lower() in ("null", "~", ""):
        return None
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v
