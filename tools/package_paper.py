"""Create a self-contained Overleaf source archive from the current manuscript."""
from pathlib import Path
import re
import zipfile

root=Path(__file__).resolve().parents[1]
paper=root/'paper'
text=(paper/'main.tex').read_text(encoding='utf-8')
files=[paper/name for name in ('main.tex','references.bib','cas-dc.cls',
       'cas-common.sty','cas-model2-names.bst','README.md','requirements-results.txt')]
files+=sorted((paper/'generated').glob('*.tex'))
files+=sorted((paper/'thumbnails').glob('*.jpeg'))
figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
files += [paper/'figures'/name for name in sorted(set(figures))]
missing=[str(f) for f in files if not f.is_file()]
if missing: raise FileNotFoundError(missing)
out=root/'output';out.mkdir(exist_ok=True)
target=out/'ma-cetsp-overleaf.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.relative_to(paper).as_posix())
with zipfile.ZipFile(target) as z:assert z.testzip() is None
print(f'{target}: {len(files)} files, {target.stat().st_size} bytes')
