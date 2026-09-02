"""Render a compiled paper and report figure/table locations for visual review."""
from pathlib import Path
import re
import sys
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image, ImageDraw

path=Path(sys.argv[1] if len(sys.argv)>1 else 'output/pdf/main.pdf')
out=Path('tmp/paper-review');out.mkdir(parents=True,exist_ok=True)
reader=PdfReader(path); doc=pdfium.PdfDocument(path)
thumbs=[]
for i,page in enumerate(reader.pages):
    t=page.extract_text()
    labels=re.findall(r'(?:Figure|Table)\s+\d+\s*[:.]',t)
    print(f'page {i+1:02}: {len(t):5} chars; '+', '.join(labels))
    im=doc[i].render(scale=0.85).to_pil().convert('RGB')
    draw=ImageDraw.Draw(im);draw.text((8,8),f'Page {i+1}',fill='red')
    thumbs.append(im)
    if labels or i==0:
        doc[i].render(scale=1.65).to_pil().save(out/f'page-{i+1:02}.png')
for start in range(0,len(thumbs),4):
    cells=thumbs[start:start+4];w=max(x.width for x in cells);h=max(x.height for x in cells)
    sheet=Image.new('RGB',(2*w,2*h),'#dddddd')
    for j,im in enumerate(cells):sheet.paste(im,((j%2)*w,(j//2)*h))
    sheet.save(out/f'overview-{start//4+1}.png')
print(f'{len(reader.pages)} pages; images in {out}')
