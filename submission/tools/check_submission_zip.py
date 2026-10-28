#!/usr/bin/env python3
"""Validate a Predictive Evaluation Challenge submission ZIP."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import numbers
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    from streaming_ingestion import AcquisitionHook
except ModuleNotFoundError as exc:
    if exc.name != "streaming_ingestion":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "competition_contract"))
    from streaming_ingestion import AcquisitionHook


try:
    from prediction_workers import validate_prediction_config
except ModuleNotFoundError as exc:
    if exc.name != "prediction_workers":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "competition_contract"))
    from prediction_workers import validate_prediction_config


MAX_MODELS = 5
LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
SAFE_MODELS_TXT_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)
# Keep in sync with competition_contract/submission_static_checks.py.
SENTENCE_TRANSFORMER_BASIC_MODEL_IDS = {
    "albert-base-v1",
    "albert-base-v2",
    "albert-large-v1",
    "albert-large-v2",
    "albert-xlarge-v1",
    "albert-xlarge-v2",
    "albert-xxlarge-v1",
    "albert-xxlarge-v2",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "bert-base-uncased",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking",
    "bert-large-cased",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-uncased-whole-word-masking",
    "bert-large-uncased",
    "camembert-base",
    "ctrl",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert-base-uncased",
    "distilgpt2",
    "distilroberta-base",
    "gpt2-large",
    "gpt2-medium",
    "gpt2-xl",
    "gpt2",
    "openai-gpt",
    "roberta-base-openai-detector",
    "roberta-base",
    "roberta-large-mnli",
    "roberta-large-openai-detector",
    "roberta-large",
    "t5-11b",
    "t5-3b",
    "t5-base",
    "t5-large",
    "t5-small",
    "transfo-xl-wt103",
    "xlm-clm-ende-1024",
    "xlm-clm-enfr-1024",
    "xlm-mlm-100-1280",
    "xlm-mlm-17-1280",
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-roberta-base",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    "xlm-roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased",
}
SAFE_ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".csv",
    ".joblib",
    ".json",
    ".model",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".txt",
    ".ubj",
    ".yaml",
    ".yml",
}
LOAD_CALL_SUFFIXES = (
    "open",
    ".open",
    ".read_csv",
    ".read_parquet",
    ".read_pickle",
    ".read_json",
    ".read_excel",
    ".load",
    ".loadtxt",
    ".genfromtxt",
    ".load_model",
    ".read_text",
    ".read_bytes",
    ".with_name",
)
SMOKE_SUBJECT = {
    "normalized_name": "sample-model",
    "provider": "sample-org",
    "release_date": "2026-01-01",
    "access_date": "2026-02-01",
    "harness": "sample-harness",
    "reasoning_effort": "medium",
    "harness_version": "1.0",
    "subject_features_extra": "",
}
SMOKE_ITEM = {
    "item_content": "A sample yes/no evaluation item.",
    "item_features": "tier=near-term",
    "interactors": "",
}


def _smoke_input() -> list:
    return [dict(SMOKE_SUBJECT), dict(SMOKE_ITEM)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_zip", type=Path)
    args = parser.parse_args()

    try:
        validate_submission_zip(args.submission_zip)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {args.submission_zip} looks like a valid submission ZIP.")
    return 0


def validate_submission_zip(zip_path: Path) -> None:
    if not zip_path.exists():
        raise ValueError(f"ZIP not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid ZIP file: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        normalized = {name.replace("\\", "/") for name in names}
        _reject_unsafe_members(normalized)

        if any(name.lower().endswith(".zip") for name in normalized):
            raise ValueError("Do not upload a ZIP that contains another ZIP. Upload the submission files directly.")

        if "model.py" not in normalized:
            nested_model = sorted(name for name in normalized if name.endswith("/model.py"))
            if nested_model:
                raise ValueError(
                    "model.py is nested inside a folder. Zip the contents of your submission directory, "
                    "not the directory itself."
                )
            raise ValueError("model.py must be at the ZIP root.")

        models_txt_paths = [name for name in normalized if name == "models.txt" or name.endswith("/models.txt")]
        if any(name != "models.txt" for name in models_txt_paths):
            raise ValueError("models.txt is nested inside a folder. It must be at the ZIP root.")
        if len(models_txt_paths) > 1:
            raise ValueError("Only one models.txt file is allowed, at the ZIP root.")

        models = _read_models_txt(zf, normalized)
        if "prediction_config.json" in normalized:
            try:
                validate_prediction_config(json.loads(zf.read("prediction_config.json")))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid prediction_config.json: {exc}") from exc
        if len(models) > MAX_MODELS:
            raise ValueError(f"models.txt lists {len(models)} models; maximum allowed is {MAX_MODELS}.")
        _validate_models_txt_entries(models)

        with tempfile.TemporaryDirectory(prefix="submission-check-") as tmpdir:
            zf.extractall(tmpdir)
            submission_dir = Path(tmpdir)
            _check_requirements(submission_dir)
            _check_missing_local_artifacts(submission_dir)
            _check_declared_huggingface_model_references(submission_dir, set(models))
            _check_model(submission_dir)
            _check_labeling(submission_dir)


def _reject_unsafe_members(names: set[str]) -> None:
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("ZIP contains an unsafe file path. Recreate it from the submission directory contents.")


def _read_models_txt(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "models.txt" not in names:
        return []
    with zf.open("models.txt") as handle:
        text = handle.read().decode("utf-8", errors="replace")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _validate_models_txt_entries(models: list[str]) -> None:
    seen: set[str] = set()
    for model_id in models:
        if model_id in seen:
            raise ValueError(f"models.txt contains duplicate entry {model_id!r}.")
        seen.add(model_id)
        if (
            not SAFE_MODELS_TXT_RE.fullmatch(model_id)
            or ".." in model_id
            or re.search(r"^(?:hf|sk|ghp|gho|github_pat)[_-]", model_id, re.IGNORECASE)
            or any(re.search(r"^(?:hf|sk|ghp|gho|github_pat)[_-]", part, re.IGNORECASE) for part in model_id.split("/"))
        ):
            raise ValueError(
                "models.txt entries must be plain HuggingFace repo IDs, not URLs, tokens, local paths, or commands."
            )


def _check_missing_local_artifacts(submission_dir: Path) -> None:
    for source_path in _runtime_python_files(submission_dir):
        try:
            source = source_path.read_text(errors="replace")
            tree = ast.parse(source, filename=source_path.name)
        except (OSError, SyntaxError):
            continue
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func) or ""
            if not _call_may_load_local_file(call_name):
                continue
            if _call_is_write_open(call_name, node, constants):
                continue
            rel_source_path = source_path.relative_to(submission_dir)
            for value in _literal_call_strings(node, constants, rel_source_path):
                missing = _missing_artifact_path(submission_dir, value)
                if missing:
                    rel_source = rel_source_path.as_posix()
                    artifact_name = Path(missing).name
                    raise ValueError(
                        f"{rel_source}:{node.lineno} references bundled file {artifact_name!r}, "
                        "but it is not in the ZIP. Add the file or update the relative path."
                    )


def _runtime_python_files(submission_dir: Path) -> list[Path]:
    parsed: dict[Path, ast.Module] = {}
    module_to_path: dict[str, Path] = {}
    for path in sorted(submission_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"), filename=path.name)
        except (OSError, SyntaxError):
            continue
        parsed[path] = tree
        relpath = path.relative_to(submission_dir).with_suffix("")
        parts = list(relpath.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            module_to_path[".".join(parts)] = path

    entrypoints = [submission_dir / name for name in ("model.py", "labeling.py")]
    selected: list[Path] = []
    stack = [path for path in entrypoints if path in parsed]
    while stack:
        path = stack.pop()
        if path in selected:
            continue
        selected.append(path)
        tree = parsed[path]
        relpath = path.relative_to(submission_dir)
        for module_name in _local_import_candidates(tree, relpath):
            imported = module_to_path.get(module_name)
            if imported and imported not in selected:
                stack.append(imported)
    return selected or [path for path in entrypoints if path.exists()]


def _local_import_candidates(tree: ast.Module, relpath: Path) -> set[str]:
    visitor = _LocalImportCandidateVisitor(relpath)
    visitor.visit(tree)
    return visitor.candidates


class _LocalImportCandidateVisitor(ast.NodeVisitor):
    def __init__(self, relpath: Path):
        self.candidates: set[str] = set()
        current_module = ".".join(relpath.with_suffix("").parts)
        if current_module.endswith(".__init__"):
            self.current_package = current_module.rsplit(".", 1)[0]
        else:
            self.current_package = current_module.rsplit(".", 1)[0] if "." in current_module else ""

    def visit_If(self, node: ast.If) -> None:
        if _is_main_guard(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.candidates.add(alias.name)
            self.candidates.add(alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = self.current_package.split(".") if self.current_package else []
            if node.level > len(base) + 1:
                return
            prefix_parts = base[:len(base) - node.level + 1]
            if node.module:
                prefix_parts.extend(node.module.split("."))
            prefix = ".".join(part for part in prefix_parts if part)
        else:
            prefix = node.module or ""
        if prefix:
            self.candidates.add(prefix)
        for alias in node.names:
            if prefix:
                self.candidates.add(f"{prefix}.{alias.name}")
            elif alias.name != "*":
                self.candidates.add(alias.name)


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        else:
            continue
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants[target.id] = value.value
        elif isinstance(target, ast.Name):
            constants.pop(target.id, None)
    return constants


def _call_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    if parts:
        return ".".join(reversed(parts))
    return None


def _call_may_load_local_file(call_name: str) -> bool:
    if call_name in {"open", "load_model", "read_text", "read_bytes"}:
        return True
    return any(call_name.endswith(suffix) for suffix in LOAD_CALL_SUFFIXES)


def _call_is_write_open(call_name: str, node: ast.Call, constants: dict[str, str]) -> bool:
    if not (call_name == "open" or call_name.endswith(".open")):
        return False
    mode = ""
    if len(node.args) >= 2:
        mode = _string_literal(node.args[1], constants) or ""
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode = _string_literal(keyword.value, constants) or mode
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _literal_call_strings(
    node: ast.Call,
    constants: dict[str, str],
    source_path: Path | None = None,
) -> list[str]:
    values: list[str] = []
    for arg in node.args[:2]:
        literal = _path_literal(arg, constants, source_path)
        if literal is not None:
            values.append(literal)
    for keyword in node.keywords:
        if keyword.arg in {"path", "filepath", "filename", "file", "fname"}:
            literal = _path_literal(keyword.value, constants, source_path)
            if literal is not None:
                values.append(literal)
    if isinstance(node.func, ast.Attribute):
        literal = _path_literal(node.func.value, constants, source_path)
        if literal is not None:
            values.append(literal)
    return values


def _string_literal(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value or None
    if isinstance(node, ast.Name):
        return constants.get(node.id) or None
    return None


def _path_literal(
    node: ast.AST,
    constants: dict[str, str],
    source_path: Path | None = None,
) -> str | None:
    literal = _string_literal(node, constants)
    if literal is not None:
        return literal
    if isinstance(node, ast.Call):
        call_name = _call_name(node.func) or ""
        if call_name in {"Path", "pathlib.Path"} and node.args:
            return _path_literal(node.args[0], constants, source_path)
        if call_name in {"os.path.join", "posixpath.join", "ntpath.join"}:
            parts: list[str] = []
            for arg in node.args:
                part = _path_literal(arg, constants, source_path)
                if part is None:
                    return None
                parts.append(part)
            return os.path.join(*parts) if parts else None
        if call_name.endswith(".with_name") and node.args:
            return _source_relative_with_name(
                node.func,
                _path_literal(node.args[0], constants, source_path),
                source_path,
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "with_name" and node.args:
            return _source_relative_with_name(
                node.func,
                _path_literal(node.args[0], constants, source_path),
                source_path,
            )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _path_literal(node.left, constants, source_path)
        right = _path_literal(node.right, constants, source_path)
        if left is not None and right is not None:
            return left + right
    return None


def _source_relative_with_name(
    func_node: ast.AST,
    filename: str | None,
    source_path: Path | None,
) -> str | None:
    if filename is None:
        return None
    if not isinstance(func_node, ast.Attribute):
        return filename
    if source_path is None or not _path_expr_is_dunder_file(func_node.value):
        return filename
    source_dir = source_path.parent
    return (source_dir / filename).as_posix() if source_dir.parts else filename


def _path_expr_is_dunder_file(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "__file__":
        return True
    if isinstance(node, ast.Call):
        call_name = _call_name(node.func) or ""
        if call_name in {"Path", "pathlib.Path"} and node.args:
            return _path_expr_is_dunder_file(node.args[0])
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "absolute"}:
            return _path_expr_is_dunder_file(node.func.value)
    return False


def _missing_artifact_path(submission_dir: Path, value: str) -> str:
    value = (value or "").strip()
    if not value or "://" in value or value.startswith(("~", "$")):
        return ""
    if any(marker in value for marker in ("{", "}", "*", "?")):
        return ""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return ""
    if path.name in {"requirements.txt", "models.txt"}:
        return ""
    if re.search(
        r"secret|token|password|hidden|source[_-]?item|item[_-]?id|"
        r"source[_-]?id|label[_-]?id|ground[_-]?truth|answer",
        path.name,
        re.IGNORECASE,
    ):
        return ""
    if path.suffix.lower() not in SAFE_ARTIFACT_SUFFIXES:
        return ""
    candidate = (submission_dir / path).resolve()
    try:
        candidate.relative_to(submission_dir.resolve())
    except ValueError:
        return ""
    if candidate.exists():
        return ""
    return path.as_posix()


def _check_requirements(submission_dir: Path) -> None:
    requirements = submission_dir / "requirements.txt"
    nested = sorted(
        path.relative_to(submission_dir).as_posix()
        for path in submission_dir.rglob("requirements.txt")
        if path != requirements
    )
    if nested:
        raise ValueError("requirements.txt must be at the ZIP root, not inside a folder.")
    if not requirements.exists():
        return
    for lineno, raw_line in enumerate(requirements.read_text(errors="replace").splitlines(), start=1):
        line = re.sub(r"\s+#.*$", "", raw_line.strip()).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or "://" in line or line.startswith(("./", "../", "/")):
            raise ValueError(
                f"requirements.txt line {lineno} is unsupported. Use named pip packages only; "
                "pip option lines, URLs, local paths, and nested requirements files are not supported."
            )
        if not re.match(
            r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[^\]]+\])?(?:\s*(?:===|==|~=|!=|<=|>=|<|>|;).*)?$",
            line,
        ):
            raise ValueError(
                f"requirements.txt line {lineno} is unsupported. Use named pip packages only."
            )


def _check_declared_huggingface_model_references(submission_dir: Path, declared_models: set[str]) -> None:
    refs: list[tuple[str, int, str]] = []
    for source_path in _runtime_python_files(submission_dir):
        try:
            tree = ast.parse(source_path.read_text(errors="replace"), filename=source_path.name)
        except (OSError, SyntaxError):
            continue
        constants = _module_string_constants(tree)
        rel_source = source_path.relative_to(submission_dir).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func) or ""
            if not (
                call_name.endswith(".from_pretrained")
                or call_name == "SentenceTransformer"
                or call_name.endswith(".SentenceTransformer")
            ):
                continue
            model_ref = _pretrained_model_reference(node, constants)
            if model_ref is None or _is_local_model_reference(submission_dir, model_ref):
                continue
            declared_ref = _declared_model_ref_for_call(model_ref, call_name)
            if declared_ref not in declared_models:
                refs.append((rel_source, node.lineno, declared_ref))
    if refs:
        rel_source, lineno, model_ref = refs[0]
        raise ValueError(
            "HuggingFace model references must be listed in models.txt or point to local files "
            f"bundled in the ZIP. Missing {model_ref!r} used at {rel_source}:{lineno}."
        )


def _pretrained_model_reference(node: ast.Call, constants: dict[str, str]) -> str | None:
    if node.args:
        literal = _string_literal(node.args[0], constants)
        if literal is not None:
            return literal
    for keyword in node.keywords:
        if keyword.arg in {
            "pretrained_model_name_or_path",
            "model_name_or_path",
            "model_name",
            "model",
            "model_id",
            "repo_id",
            "path",
        }:
            literal = _string_literal(keyword.value, constants)
            if literal is not None:
                return literal
    return None


def _is_local_model_reference(submission_dir: Path, model_ref: str) -> bool:
    if not model_ref or os.path.isabs(model_ref):
        return False
    if "://" in model_ref or model_ref.startswith(("~", "$")):
        return False
    candidate = (submission_dir / model_ref).resolve()
    try:
        candidate.relative_to(submission_dir.resolve())
    except ValueError:
        return False
    return candidate.exists()


def _declared_model_ref_for_call(model_ref: str, call_name: str) -> str:
    if (
        "SentenceTransformer" in call_name
        and "/" not in model_ref
        and model_ref.lower() not in SENTENCE_TRANSFORMER_BASIC_MODEL_IDS
    ):
        return f"sentence-transformers/{model_ref}"
    return model_ref


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    left, right = node.left, node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _check_model(submission_dir: Path) -> None:
    model = _load_module(submission_dir / "model.py", "submission_model", submission_dir)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("model.py must define callable predict(input, labeled=None).")
    try:
        value = predict(_smoke_input(), labeled=[])
    except TypeError as exc:
        raise ValueError("predict() must accept the labeled keyword argument.") from exc
    except Exception as exc:
        raise ValueError("predict() raised during the local smoke check.") from exc
    _assert_finite_probability(value, "predict()")


def _check_labeling(submission_dir: Path) -> None:
    labeling_path = submission_dir / "labeling.py"
    if not labeling_path.exists():
        return
    labeling = _load_module(labeling_path, "submission_labeling", submission_dir)
    acquisition = getattr(labeling, "acquisition_function", None)
    if not callable(acquisition):
        raise ValueError("labeling.py must define callable acquisition_function(input, prediction, labeled, context).")
    try:
        value = AcquisitionHook(acquisition)(_smoke_input(), 0.5, [], {
            "subject_id": "subject_001", "benchmark_id": "benchmark_001",
            "labels_remaining": 31, "labels_acquired": 0, "max_labels": 31,
            "items_remaining": 40,
        })
    except Exception as exc:
        raise ValueError("acquisition_function() raised during the local smoke check.") from exc


def _load_module(path: Path, module_name: str, submission_dir: Path):
    previous_path = list(sys.path)
    previous_env = os.environ.get(LOCAL_SMOKE_TEST_ENV)
    sys.path.insert(0, str(submission_dir))
    os.environ[LOCAL_SMOKE_TEST_ENV] = "1"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import {path.name}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_env is None:
            os.environ.pop(LOCAL_SMOKE_TEST_ENV, None)
        else:
            os.environ[LOCAL_SMOKE_TEST_ENV] = previous_env
        sys.path[:] = previous_path


def _assert_finite_probability(value, label: str) -> None:
    number = _assert_finite_number(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must return a probability in [0, 1].")


def _assert_finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must return a finite numeric value.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must return a finite numeric value.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must return a finite numeric value.")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
