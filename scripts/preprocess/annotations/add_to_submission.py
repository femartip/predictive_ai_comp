"""Add the ADeLe annotator (item_annotation.py) to a submission zip.

    python scripts/preprocess/annotations/add_to_submission.py <submission.zip> [--output <new.zip>]

Writes a copy of the zip (default: <name>_adele.zip next to it) with:
- item_annotation.py, adele_prompt.py and rubrics/adele_v1/*.txt at the root (replacing older copies);
- the base model and the adapter in models.txt (at most 5 repositories in all);
- in model.py, the annotation after the subject step: annotate(*process(*preprocess(input, labeled))). If model.py
  does not have that call, the zip is written without touching it and the line to add is printed.
"""

import argparse
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
MAX_MODELS = 5
FILES = {'item_annotation.py': HERE / 'item_annotation.py', 'adele_prompt.py': HERE / 'adele_prompt.py',
         **{f'rubrics/adele_v1/{f.name}': f for f in sorted((HERE / 'rubrics' / 'adele_v1').glob('*.txt'))}}
REPOS = ['Qwen/Qwen2.5-7B-Instruct', 'ceferra/qwen2.5-7b-adele-annotator']
IMPORT, NEW_IMPORT = 'from subject_processing import process\n', \
    'from subject_processing import process\nfrom item_annotation import annotate\n'
CALL, NEW_CALL = 'process(*preprocess(input, labeled))', 'annotate(*process(*preprocess(input, labeled)))'


def models_txt(text):
    repos = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    repos += [r for r in REPOS if r not in repos]
    if len(repos) > MAX_MODELS:
        raise SystemExit(f'models.txt would list {len(repos)} repositories, at most {MAX_MODELS} allowed: {repos}')
    return '\n'.join(repos) + '\n'


def model_py(text):
    if 'item_annotation' in text:
        return text, True
    if IMPORT not in text or text.count(CALL) != 1:
        return text, False
    return text.replace(IMPORT, NEW_IMPORT).replace(CALL, NEW_CALL), True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('zip', type=Path)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    out = a.output or a.zip.with_name(a.zip.stem + '_adele.zip')
    if out.resolve() == a.zip.resolve():
        raise SystemExit('--output must differ from the input zip')
    patched = True
    tmp = out.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(a.zip) as src, zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as dst:
            names = src.namelist()
            for name in names:
                if name in FILES or name.startswith('rubrics/adele_v1/') or name.endswith('/'):
                    continue
                data = src.read(name)
                if name == 'models.txt':
                    data = models_txt(data.decode()).encode()
                elif name == 'model.py':
                    text, patched = model_py(data.decode())
                    data = text.encode()
                dst.writestr(src.getinfo(name), data)
            if 'models.txt' not in names:
                dst.writestr('models.txt', models_txt(''))
            for name, path in FILES.items():
                dst.write(path, name)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f'wrote {out} ({len(FILES)} annotator files added)')
    if not patched:
        print(f'model.py has no single "{CALL}": add the annotation step by hand:\n'
              f'    {NEW_IMPORT.splitlines()[1]}\n    ... = {NEW_CALL}')


if __name__ == '__main__':
    main()
