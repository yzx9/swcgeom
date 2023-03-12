"""Base SWC format utils."""


import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["get_dsu"]


def get_dsu(df: pd.DataFrame) -> npt.NDArray[np.int32]:
    """Get disjoint set union."""
    dsu = np.where(df["pid"] == -1, df["id"], df["pid"])  # Disjoint Set Union

    id2idx = dict(zip(df["id"], range(len(df))))
    dsu = np.array([id2idx[i] for i in dsu], dtype=np.int32)

    while True:
        flag = True
        for i, p in enumerate(dsu):
            if dsu[i] != dsu[p]:
                dsu[i] = dsu[p]
                flag = False

        if flag:
            break

    return dsu
