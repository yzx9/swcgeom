"""Check common """

import warnings

import numpy as np
import pandas as pd

from .base import get_dsu

__all__ = ["check_single_root", "is_single_root"]


def check_single_root(*args, **kwargs) -> bool:
    warnings.warn(
        "`check_single_root` has been renamed to `is_single_root` since"
        "v0.5.0, and will be removed in next version",
        DeprecationWarning,
    )
    return is_single_root(*args, **kwargs)


def is_single_root(df: pd.DataFrame) -> bool:
    """Check is it only one root."""
    return len(np.unique(get_dsu(df))) == 1
