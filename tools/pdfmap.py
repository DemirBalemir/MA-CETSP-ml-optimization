"""Where did each float actually land? Maps captions and section headings to pages."""
import re
import sys
from pypdf import PdfReader

r = PdfReader(sys.argv[1])
print(f"sayfa sayisi: {len(r.pages)}\n")

sec = re.compile(r"^\s*(\d+(?:\.\d+)?)\.?\s+([A-Z][A-Za-z].{3,60})$")
cap = re.compile(r"(Figure|Table)\s+(\d+)[.:]?\s*(.{0,55})", re.S)

for i, page in enumerate(r.pages, 1):
    try:
        t = page.extract_text() or ""
    except Exception:
        continue
    hits = []
    for m in cap.finditer(t):
        label = f"{m.group(1)} {m.group(2)}"
        if label not in [h[0] for h in hits]:
            hits.append((label, " ".join(m.group(3).split())[:45]))
    heads = []
    for line in t.split("\n"):
        m = sec.match(line.strip())
        if m and len(m.group(2).split()) <= 7:
            heads.append(f"{m.group(1)} {m.group(2).strip()}")
    if hits or heads:
        print(f"--- s.{i} ---")
        for h in heads[:3]:
            print(f"    BOLUM  {h}")
        for label, txt in hits:
            print(f"    {label:<9} {txt}")
