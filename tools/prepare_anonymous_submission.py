"""Prepare a separate, self-contained anonymous manuscript; preserve the master."""
from pathlib import Path
import re
import shutil
import sys
import zipfile
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
DEST = ROOT / 'output/submission/anonymous'
BUILD = ROOT / 'tmp/paper-build/anonymous'
PDF = ROOT / 'output/pdf/manuscript_anonymous.pdf'

def without_comments(text):
    lines = []
    for line in text.splitlines():
        for i, char in enumerate(line):
            if char == '%':
                n = len(line[:i]) - len(line[:i].rstrip('\\'))
                if n % 2 == 0:
                    line = line[:i]
                    break
        lines.append(line.rstrip())
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)) + '\n'

def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

def clean_pdf(source, target):
    reader = PdfReader(source)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.metadata = None
    writer.add_metadata({'/Title': 'Pre-local-search surrogate filtering', '/Author': ''})
    writer._root_object.pop('/Metadata', None)
    with target.open('wb') as stream:
        writer.write(stream)

def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    text = (PAPER / 'main.tex').read_text(encoding='utf8')
    text = replace_once(text, r'\shortauthors{D. Balemir and D. Cant{\"u}rk}',
                        r'\shortauthors{Anonymous manuscript}')
    start = text.index(r'\author[1]')
    end = text.index(r'\cortext[1]{Corresponding author.}') + len(r'\cortext[1]{Corresponding author.}')
    text = text[:start] + text[end:]
    text = replace_once(text,
        'that selective filtering might save. In a preliminary\n'
        'study~\\citep{balemir2026lacetsp} we augmented MA-CETSP with a\n'
        'survival-analysis-based offspring filter and reported that 13--26\\% of offspring',
        'that selective filtering might save. A preliminary\n'
        'study~\\citep{balemir2026lacetsp} augmented MA-CETSP with a\n'
        'survival-analysis-based offspring filter and reported that 13--26\\% of offspring')
    text = replace_once(text,
        'In our preliminary\nstudy~\\citep{balemir2026lacetsp} we introduced a learning-assisted extension of',
        'The preliminary\nstudy~\\citep{balemir2026lacetsp} introduced a learning-assisted extension of')
    text = replace_once(text, 'filter, including our own earlier one.',
                        'filter, including the one in the preliminary study.')
    text = replace_once(text, 'retrospectively to our own project.',
                        'retrospectively to the system studied here.')
    start = text.index(r'\section*{CRediT authorship contribution statement}')
    end = text.index(r'\section*{Declaration of competing interest}', start)
    credit = without_comments(text[start:end])
    text = text[:start] + text[end:]
    start = text.index(r'\section*{Data availability}')
    end = text.index(r'\section*{Declaration of generative AI', start)
    original_data = without_comments(text[start:end])
    anonymous_data = r'''\section*{Data availability}
The solver, survival-modelling pipeline, diagnostic scripts, retained measurement
CSVs and parsed end-to-end results are available in a public repository. Its
identifying link is omitted from this manuscript for double-anonymized review;
repository details are provided separately to the editor. The release rebuilds
tables and figures from retained measurements and documents the evaluation
protocols. Raw per-solution logs, approximately 3.4 million records, are available
on request through the editor for recalculating diagnostics. The full-VND
calibration includes a run manifest and input hashes; exact historical selections
for every older probe are not bundled. The repository documentation distinguishes
result-level reproduction from refitting models on the original logs.

'''
    text = text[:start] + anonymous_data + text[end:]
    text = without_comments(text)
    # Clear identifiers after CAS has populated document metadata at maketitle.
    text = replace_once(text, r'\maketitle',
        r'\renewcommand{\printorcid}{}' + '\n' + r'\maketitle' + '\n' +
        r'\hypersetup{pdfauthor={},pdfsubject={},pdfkeywords={}}')
    forbidden = ('Demir', 'Deniz', 'tedu.edu.tr', 'TED University', '0009-0005',
                 '0000-0002-7606', 'github.com', 'our preliminary', 'our own earlier')
    assert not [s for s in forbidden if s.lower() in text.lower()]
    (DEST / 'main.tex').write_text(text, encoding='utf8')
    (DEST / 'references.bib').write_text(
        without_comments((PAPER / 'references.bib').read_text(encoding='utf8')), encoding='utf8')
    for name in ('cas-dc.cls', 'cas-common.sty', 'cas-model2-names.bst'):
        shutil.copy2(PAPER / name, DEST / name)
    (DEST / 'generated').mkdir(exist_ok=True)
    for source in sorted((PAPER / 'generated').glob('*.tex')):
        (DEST / 'generated' / source.name).write_text(
            without_comments(source.read_text(encoding='utf8')), encoding='utf8')
    (DEST / 'thumbnails').mkdir(exist_ok=True)
    for source in (PAPER / 'thumbnails').glob('*.jpeg'):
        shutil.copy2(source, DEST / 'thumbnails' / source.name)
    (DEST / 'figures').mkdir(exist_ok=True)
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text)
    for name in sorted(set(figures)):
        clean_pdf(PAPER / 'figures' / name, DEST / 'figures' / name)
    (DEST / 'README.txt').write_text(
        'Anonymous LaTeX manuscript\n\n'
        'Main file: main.tex. APA 7 references require biblatex-apa and Biber.\n'
        'Compile with latexmk -pdf main.tex (Biber is detected automatically).\n'
        'All referenced figures and generated TeX inputs are included.\n'
        'Figure 1 is editable TikZ within main.tex.\n'
        'Bibliographic author names are retained as ordinary scientific citations.\n', encoding='utf8')
    editor = PAPER / 'submission/editor_information.txt'
    editor.parent.mkdir(exist_ok=True)
    editor.write_text(
        'EDITOR ONLY — exclude from reviewer-facing files\n\n'
        'This file records information omitted from the anonymous manuscript.\n'
        'Supply it to the editor in a file category hidden from reviewers.\n'
        'It is not included in the anonymous source ZIP.\n\n'
        'Corresponding author: Demir Balemir, demir.balemir@tedu.edu.tr\n'
        'Co-author: Deniz Cantürk, deniz.canturk@tedu.edu.tr\n\n'
        'Public repository: https://github.com/DemirBalemir/MA-CETSP-ml-optimization\n\n'
        'Original CRediT text (LaTeX):\n' + credit + '\n'
        'Original data-availability text (LaTeX):\n' + original_data, encoding='utf8')
    archive = ROOT / 'output/submission/manuscript_anonymous_sources.zip'
    files = sorted(p for p in DEST.rglob('*') if p.is_file())
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any('editor_information' in n or 'cover_letter' in n for n in z.namelist())
    print(f'Prepared {DEST}; {len(files)} source files in {archive}')
    print(f'Editor-only information: {editor}')

def finish():
    PDF.parent.mkdir(parents=True, exist_ok=True)
    clean_pdf(BUILD / 'main.pdf', PDF)
    reader = PdfReader(PDF)
    pages = [p.extract_text() for p in reader.pages]
    full = '\n'.join(pages)
    for token in ('Demir', 'Deniz', 'tedu.edu.tr', 'TED University', 'ORCID',
                  'github.com/Demir', 'D. Balemir and D. Cant'):
        assert token.lower() not in full.lower(), token
    assert '??' not in full
    assert not reader.metadata.get('/Author')
    uris = []
    for page in reader.pages:
        for annotation in page.get('/Annots', []):
            action = annotation.get_object().get('/A', {})
            if action.get('/URI'):
                uris.append(str(action['/URI']))
    assert not any('demir' in u.lower() or 'tedu.edu.tr' in u.lower() for u in uris)
    print(f'Anonymous PDF: {PDF}; {len(pages)} pages; identifying text, metadata and link checks passed.')

if __name__ == '__main__':
    finish() if '--finish' in sys.argv else prepare()
