"""Check common """

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import SWCNames, Topology, get_dsu, get_names

__all__ = ["is_single_root", "is_binary_tree", "check_single_root", "is_bifurcate"]


def is_single_root(df: pd.DataFrame, *, names: Optional[SWCNames] = None) -> bool:
    """Check is it only one root."""
    return len(np.unique(get_dsu(df, names=names))) == 1


def is_bifurcate(topology: Topology, *, exclude_root: bool = True) -> bool:
    """Check is it a bifurcate topology."""

    children = {}
    for idx, pid in zip(*topology):
        s = children.get(pid, [])
        s.append(idx)
        children[pid] = s

    root = children.get(-1, [])
    for k, v in children.items():
        if len(v) > 1 and (not exclude_root or k in root):
            return False

    return True


def check_single_root(*args, **kwargs) -> bool:
    warnings.warn(
        "`check_single_root` has been renamed to `is_single_root` since"
        "v0.5.0, and will be removed in next version",
        DeprecationWarning,
    )
    return is_single_root(*args, **kwargs)


def is_binary_tree(
    df: pd.DataFrame, exclude_root: bool = True, *, names: Optional[SWCNames] = None
) -> bool:
    """Check is it a binary tree."""
    warnings.warn(
        "`is_binary_tree` has been replaced by to `is_bifurcate` since"
        "v0.8.0, and will be removed in next version",
        DeprecationWarning,
    )
    names = get_names(names)
    topo = (df[names.id].to_numpy(), df[names.pid].to_numpy())
    return is_bifurcate(topo, exclude_root=exclude_root)
