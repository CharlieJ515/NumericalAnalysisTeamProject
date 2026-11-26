import argparse
import ast
import json
import re
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
SRC_MODULES: tuple[str, ...] = (
    "typedefs",
    "dataio",
    "model",
    "loss",
    "optim",
    "dataloader",
    "kfold",
    "trainer",
    "plot",
    "config",
)
MAIN_PATH: Path = ROOT / "main.py"
REQUIREMENTS_PATH: Path = ROOT / "requirements.txt"
DEFAULT_CONFIG_PATH: Path = ROOT / "configs" / "baseline.toml"
OUT: Path = ROOT / "dist"

KAGGLE_ROOT: str = 'Path("/kaggle/working")'
KAGGLE_PATH_OVERRIDES: dict[str, str] = {
    "train": 'str(next(Path("/kaggle/input").rglob("train.csv")))',
    "test": 'str(next(Path("/kaggle/input").rglob("test.csv")))',
    "out": '"/kaggle/working"',
}
MAIN_GUARD: str = 'if __name__ == "__main__":'


def strip_local_imports(source: str) -> str:
    dropped: set[int] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "src":
            dropped.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    kept = [line for i, line in enumerate(source.splitlines(), start=1) if i not in dropped]

    blocks = []
    for block in "\n".join(kept).strip().split("\n\n"):
        if block.strip():
            blocks.append(block)
    return "\n\n".join(blocks)


def prepare_main(source: str) -> str:
    source, count = re.subn(
        r"^(ROOT: Path = ).*$",
        lambda match: match.group(1) + KAGGLE_ROOT,
        source,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("main.py에서 ROOT 정의를 찾지 못했습니다.")

    if source.count(MAIN_GUARD) != 1:
        raise ValueError(f"main.py에서 {MAIN_GUARD} 블록을 찾지 못했습니다.")
    source = source[: source.index(MAIN_GUARD)].rstrip()

    if "__file__" in source:
        raise ValueError("노트북에서는 __file__을 사용할 수 없습니다.")
    return source


def config_source(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    if '"""' in text:
        raise ValueError('config에 """를 사용할 수 없습니다.')

    lines = ["import tomllib", "", f'CONFIG_TOML = """\n{text.strip()}\n"""', ""]
    lines.append("CONFIG = tomllib.loads(CONFIG_TOML)")
    for key, expr in KAGGLE_PATH_OVERRIDES.items():
        lines.append(f'CONFIG["paths"]["{key}"] = {expr}')
    return "\n".join(lines)


def pip_install_source(requirements_path: Path) -> str:
    packages = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "%pip install -q " + " ".join(packages)


def code_cell(source: str) -> dict:
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


def build_notebook(config_path: Path) -> dict:
    sources = [pip_install_source(REQUIREMENTS_PATH)]

    for name in SRC_MODULES:
        module_source = (ROOT / "src" / f"{name}.py").read_text(encoding="utf-8")
        sources.append(strip_local_imports(module_source))

    main_source = MAIN_PATH.read_text(encoding="utf-8")
    sources.append(prepare_main(strip_local_imports(main_source)))
    sources.append(config_source(config_path))
    sources.append("main(CONFIG)")

    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [code_cell(source) for source in sources],
    }


def main(config_path: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebook_path = OUT / f"{config_path.stem}.ipynb"

    notebook = build_notebook(config_path)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"config={config_path}")
    print(f"cells={len(notebook['cells'])}")
    print(f"saved={notebook_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    main(args.config)
