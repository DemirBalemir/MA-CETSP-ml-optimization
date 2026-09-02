"""Static sanity checks on main.tex: TikZ node references and env balance.

This cannot prove the file compiles. It only
catches the errors that are cheap to catch: unbalanced environments, unbalanced
delimiters inside the tikzpicture, and arrows pointing at nodes that were never
declared.
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex")
s = path.read_text(encoding="utf-8")

m = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", s, re.S)
if m:
    blk = m.group(0)
    print("--- tikzpicture ---")
    print(f"  satir           : {blk.count(chr(10))}")
    for o, c, name in (("{", "}", "brace"), ("(", ")", "paren"), ("[", "]", "bracket")):
        a, b = blk.count(o), blk.count(c)
        flag = "" if a == b else "   <-- DENGESIZ"
        print(f"  {name:<15} : {a} / {b}{flag}")
    names = set(re.findall(r"\\node(?:\[[^\]]*\])?\s*\((\w+)\)", blk))
    names.update(re.findall(r"\\coordinate\s*\((\w+)\)", blk))
    print(f"  tanimli dugum   : {sorted(names)}")
    # references of the form (name) or (name.anchor)
    # Ignore math arguments such as M(x), which are not TikZ coordinates.
    geometry = re.sub(r"\$[^$]*\$", "", blk)
    refs = set(re.findall(r"\((\w+)(?:\.[\w\s]+)?\)", geometry))
    known = names | {"0", "1"}
    unknown = sorted(r for r in refs if r not in known and not r.replace(".", "").isdigit())
    print(f"  tanimsiz referans: {unknown if unknown else 'yok'}")
else:
    print("tikzpicture bulunamadi")

print("\n--- ortam dengesi (tum dosya) ---")
bad = False
for env in ("figure*", "figure", "table*", "table", "tikzpicture",
            "algorithm", "algorithmic", "tabular", "equation", "align",
            "abstract", "document"):
    b = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", s))
    e = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", s))
    flag = ""
    if b != e:
        flag = "   <-- DENGESIZ"
        bad = True
    print(f"  {env:<12} begin={b:<3} end={e:<3}{flag}")

print("\n--- referans butunlugu ---")
labels = set(re.findall(r"\\label\{([^}]*)\}", s))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", s))
missing = sorted(refs - labels)
unused = sorted(labels - refs)
print(f"  tanimsiz \\ref  : {missing if missing else 'yok'}")
print(f"  kullanilmayan label: {unused if unused else 'yok'}")

bib = path.with_name('references.bib').read_text(encoding='utf-8')
active = re.sub(r'(?m)(?<!\\)%.*$', '', s)
keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
cited = {key.strip() for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}', active)
         for key in group.split(',')}
missing_cites = sorted(cited - keys)
print(f"\nKaynakca: {len(keys)} kayit, {len(cited)} atif anahtari; eksik: {missing_cites}")
abstract = active.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
print(f"Abstract kelime (LaTeX dahil): {len(abstract.split())}")

sys.exit(1 if (bad or missing or missing_cites) else 0)
