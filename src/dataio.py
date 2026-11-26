from pathlib import Path

import numpy as np

from src.typedefs import DataVec, IdVec

TRAIN_COLUMNS = ("id", "x", "y")
TEST_COLUMNS = ("id", "x")


def _read_numeric_csv(path: Path, columns: tuple[str, ...]) -> tuple[IdVec, tuple[DataVec, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if table.size == 0:
        raise ValueError(f"CSV에 데이터 행이 없습니다: {path}")
    loaded = table.dtype.names
    if loaded is None or tuple(loaded) != columns:
        raise ValueError(
            f"{path.name}의 열은 {', '.join(columns)} 이어야 합니다. 실제 열: {loaded}"
        )

    ids = np.asarray(table[columns[0]])
    if np.issubdtype(ids.dtype, np.floating):
        if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
            raise ValueError(f"{path.name}의 id에 정수가 아닌 값이 있습니다.")
    ids = ids.astype(np.int64)

    float_columns = []
    for name in columns[1:]:
        values = np.asarray(table[name]).astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path.name}의 {name}에 NaN 또는 Inf가 있습니다.")
        float_columns.append(values)

    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path.name}에 중복된 id가 있습니다.")
    return ids, tuple(float_columns)


def load_train(
    path: Path,
) -> tuple[IdVec, DataVec, DataVec]:
    """train.csv를 읽어 (id, x, y)를 반환한다."""
    ids, (x, y) = _read_numeric_csv(path, TRAIN_COLUMNS)
    return ids, x, y


def load_test(path: Path) -> tuple[IdVec, DataVec]:
    """test.csv를 읽어 (id, x)를 반환한다. 정답 열은 없다."""
    ids, (x,) = _read_numeric_csv(path, TEST_COLUMNS)
    return ids, x


def save_submission(ids: IdVec, y_pred: DataVec, path: Path) -> None:
    """ids, y_pred를 csv 파일로 만들어 path에 저장"""
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("예측에 NaN 또는 Inf가 있어 submission.csv를 쓰지 않습니다.")
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write("id,y_pred\n")
        for sample_id, pred in zip(ids, y_pred, strict=True):
            file.write(f"{int(sample_id)},{pred:.10f}\n")
