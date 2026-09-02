"""Does each page actually carry embedded images/forms, or only caption text?

If the last pages hold captions but no XObjects, the graphics files were never
found at compile time and LaTeX typeset empty float boxes -- a different problem
from float congestion, and one no placement package can fix.
"""
import sys
from pypdf import PdfReader

r = PdfReader(sys.argv[1])
print(f"sayfa: {len(r.pages)}\n")
for i, p in enumerate(r.pages, 1):
    res = p.get("/Resources") or {}
    xo = res.get("/XObject")
    n_img = n_form = 0
    if xo is not None:
        try:
            xo = xo.get_object()
            for k in xo:
                sub = xo[k].get_object().get("/Subtype")
                if sub == "/Image":
                    n_img += 1
                elif sub == "/Form":
                    n_form += 1
        except Exception:
            pass
    try:
        txt = (p.extract_text() or "")
    except Exception:
        txt = ""
    has_cap = ("Figure" in txt)
    if n_img or n_form or has_cap:
        mark = ""
        if has_cap and not (n_img or n_form):
            mark = "   <-- caption var, gorsel YOK"
        print(f"  s.{i:<3} image={n_img:<3} form={n_form:<3} "
              f"caption={'evet' if has_cap else 'hayir'}{mark}")
