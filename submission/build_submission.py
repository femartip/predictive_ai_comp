"""Package the submission: submission/submission/ with the current shared asset handlers.

The zip holds every file of submission/submission/ at its root (hidden files, __pycache__ and the folder's own assets/
copy left out) and assets/ from scripts/preprocess/assets, the handlers measurement_db.py uses, so the submission
never runs an edited copy of them.

    poetry run python submission/build_submission.py [--output submission/submission.zip]
"""

import argparse
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / 'submission'
ASSETS = ROOT / 'scripts/preprocess/assets'


def build(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(SOURCE.rglob('*')):
                relative = path.relative_to(SOURCE)
                skip = relative.parts[0] == 'assets' or any(p.startswith('.') or p == '__pycache__'
                                                             for p in relative.parts)
                if path.is_file() and not skip:
                    archive.write(path, relative.as_posix())
            for path in sorted(ASSETS.glob('*.py')):
                archive.write(path, 'assets/' + path.name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', type=Path, default=HERE / 'submission.zip')
    print(build(parser.parse_args().output).resolve())


if __name__ == '__main__':
    main()
