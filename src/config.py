import tomllib
from pathlib import Path
from typing import Any, TypeVar

from src.dataloader import DataLoader, FullBatchLoader, MiniBatchLoader, StochasticLoader
from src.loss import Loss, MAELoss, MSELoss
from src.model import (
    ExponentialModel,
    LinearModel,
    Model,
    PolynomialModel,
    PowerModel,
    SaturationModel,
)
from src.optim import SGD, Adagrad, Adam, AdaptiveSGD, Momentum, Nesterov, Optimizer, RMSProp

T = TypeVar("T")

MODELS: dict[str, type[Model]] = {
    "LinearModel": LinearModel,
    "PolynomialModel": PolynomialModel,
    "ExponentialModel": ExponentialModel,
    "SaturationModel": SaturationModel,
    "PowerModel": PowerModel,
}
LOSSES: dict[str, type[Loss]] = {
    "MSELoss": MSELoss,
    "MAELoss": MAELoss,
}
OPTIMIZERS: dict[str, type[Optimizer]] = {
    "SGD": SGD,
    "Momentum": Momentum,
    "Nesterov": Nesterov,
    "Adagrad": Adagrad,
    "RMSProp": RMSProp,
    "Adam": Adam,
    "AdaptiveSGD": AdaptiveSGD,
}
DATALOADERS: dict[str, type[DataLoader]] = {
    "MiniBatchLoader": MiniBatchLoader,
    "FullBatchLoader": FullBatchLoader,
    "StochasticLoader": StochasticLoader,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def build(registry: dict[str, type[T]], section: dict[str, Any], **extra: Any) -> T:
    kwargs = dict(section)
    name = kwargs.pop("name")
    if name not in registry:
        raise KeyError(f"알 수 없는 이름입니다: {name} (가능: {', '.join(registry)})")

    return registry[name](**kwargs, **extra)
