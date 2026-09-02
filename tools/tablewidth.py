"""Rough overflow check: how wide is each table's widest row, in characters?

A cas-dc single column fits roughly 46-50 characters at \small. Anything wider
in a plain `table` environment will run into the neighbouring column, which is
what happened to tab:model-comparison and tab:stage-chain before they were
moved to `table*`.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex"
s = open(path, encoding="utf-8").read()

pat = re.compile(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", re.S)
for m in pat.finditer(s):
    env, body = m.group(1), m.group(2)
    lab = re.search(r"\\label\{(.*?)\}", body)
    name = lab.group(1) if lab else "?"
    inner = body.split(r"\begin{tabular}")[-1]
    widths = []
    for row in inner.split(r"\\"):
        if "&" not in row:
            continue
        clean = re.sub(r"\\[a-zA-Z]+\s*", "", row)
        clean = clean.translate(str.maketrans("", "", "{}$&"))
        widths.append(len(clean.strip()) + row.count("&") * 2)
    mx = max(widths) if widths else 0
    if env == "table" and mx > 46:
        flag = "<-- TASAR, table* yap"
    elif env == "table*":
        flag = "(tam genislik)"
    else:
        flag = "ok"
    print(f"{env:<7} {name:<24} ~{mx:>3} karakter   {flag}")
