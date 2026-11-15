"""Result files: small JSON helpers so every number in the README traces to a file."""
from __future__ import annotations

import json

import numpy as np

from . import config


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def save_json(name: str, obj) -> None:
    with open(config.RESULTS / name, "w") as f:
        json.dump(obj, f, indent=1, default=_default)


def load_json(name: str):
    with open(config.RESULTS / name) as f:
        return json.load(f)
