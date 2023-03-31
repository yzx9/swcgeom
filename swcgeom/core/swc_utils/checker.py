"""Check common """

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import SWCNames, get_dsu, get_names

__all__ = ["check_single_root", "is_single_root", "is_binary_tree"]


def check_single_root(*args, **kwargs) -> bool:
    warnings.warn(
        "`check_single_root` has been renamed to `is_single_root` since"
        "v0.5.0, and will be removed in next version",
        DeprecationWarning,
    )
    return is_single_root(*args, **kwargs)


def is_single_root(df: pd.DataFrame, *, names: Optional[SWCNames] = None) -> bool:
    """Check is it only one root."""
    return len(np.unique(get_dsu(df, names=names))) == 1


def is_binary_tree(
    df: pd.DataFrame, exclude_root: bool = True, *, names: Optional[SWCNames] = None
) -> bool:
    """Check is it a binary tree."""
    names = get_names(names)

    children = {}
    for idx, pid in zip(df[names.id], df[names.pid]):
        s = children.get(pid, [])
        s.append(idx)
        children[pid] = s

    root = children.get(-1, [])
    for k, v in children.items():
        if len(v) > 1 and (not exclude_root or k in root):
            return False

    return True
