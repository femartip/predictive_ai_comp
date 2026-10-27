"""Package the data-processing contribution with the current shared asset handlers."""

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def build(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((ROOT / 'submission').iterdir()):
                if path.is_file() and not path.name.startswith('.'):
                    archive.write(path, path.name)
            for path in sorted((ROOT / 'scripts/preprocess/assets').glob('*.py')):
                archive.write(path, 'assets/' + path.name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data_processing_submission.zip')
    print(build(parser.parse_args().output).resolve())


if __name__ == '__main__':
    main()
