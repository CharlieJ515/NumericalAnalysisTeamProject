import json
import re
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
SRC_MODULES: tuple[str, ...] = ("typedefs", "dataio", "model", "optim", "plot")
MAIN_PATH: Path = ROOT / "main.py"
REQUIREMENTS_PATH: Path = ROOT / "requirements.txt"
OUT: Path = ROOT / "dist"
NOTEBOOK_PATH: Path = OUT / "kaggle.ipynb"

LOCAL_ONLY_NAMES: tuple[str, ...] = ("ROOT",)
KAGGLE_PATHS: dict[str, str] = {
    "TRAIN_PATH": 'next(Path("/kaggle/input").rglob("train.csv"))',
    "TEST_PATH": 'next(Path("/kaggle/input").rglob("test.csv"))',
    "OUT": 'Path("/kaggle/working")',
}


def strip_local_imports(source: str) -> str:
    blocks = []
    for block in source.strip().split("\n\n"):
        lines = [line for line in block.split("\n") if not line.startswith("from src.")]
        if any(line.strip() for line in lines):
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def use_kaggle_paths(source: str) -> str:
    for name in LOCAL_ONLY_NAMES:
        source, count = re.subn(rf"^{name}: Path = .*\n", "", source, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"main.py에서 {name} 정의를 찾지 못했습니다.")

    for name, expr in KAGGLE_PATHS.items():
        source, count = re.subn(
            rf"^({name}: Path = ).*$",
            lambda match, expr=expr: match.group(1) + expr,
            source,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"main.py에서 {name} 정의를 찾지 못했습니다.")

    if "__file__" in source:
        raise ValueError("노트북에서는 __file__을 사용할 수 없습니다.")
    return source


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


def build_notebook() -> dict:
    sources = [pip_install_source(REQUIREMENTS_PATH)]

    for name in SRC_MODULES:
        module_source = (ROOT / "src" / f"{name}.py").read_text(encoding="utf-8")
        sources.append(strip_local_imports(module_source))

    main_source = MAIN_PATH.read_text(encoding="utf-8")
    sources.append(use_kaggle_paths(strip_local_imports(main_source)))

    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [code_cell(source) for source in sources],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"cells={len(notebook['cells'])}")
    print(f"saved={NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
