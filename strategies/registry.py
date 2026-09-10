from strategies.ai_strategy import AIStrategy
from strategies.base import BaseStrategy
from strategies.breakout import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy

_registry = {}


def register(cls):
    _registry[cls.name] = cls
    return cls


@register
class _M(MomentumStrategy):
    pass


@register
class _MR(MeanReversionStrategy):
    pass


@register
class _B(BreakoutStrategy):
    pass


@register
class _AI(AIStrategy):
    pass


def get_strategy_class(name: str):
    if name not in _registry:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(_registry.keys())}")
    return _registry[name]


def create_strategy(name: str, params: dict = None) -> BaseStrategy:
    return get_strategy_class(name)(params)


def list_strategies() -> list:
    return [
        {
            "name": cls.name,
            "description": cls.description,
            "default_params": cls.default_params,
        }
        for cls in _registry.values()
    ]


def strategy_names() -> list:
    return list(_registry.keys())


# Register built-in strategies explicitly (guarantees registry is non-empty)
for _cls in [MomentumStrategy, MeanReversionStrategy, BreakoutStrategy]:
    if _cls.name not in _registry:
        register(_cls)