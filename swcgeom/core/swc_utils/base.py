"""Base SWC format utils."""

from typing import Callable, List, Literal, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["Topology", "get_dsu", "traverse"]

T, K = TypeVar("T"), TypeVar("K")

Topology = Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]  # (id, pid)


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


# fmt: off
@overload
def traverse(topology: Topology, *, enter: Callable[[int, T | None], T], root: int | np.integer = ..., mode: Literal["dfs"] = ...) -> None: ...
@overload
def traverse(topology: Topology, *, leave: Callable[[int, List[K]], K], root: int | np.integer = ..., mode: Literal["dfs"] = ...) -> K: ...
@overload
def traverse(
    topology: Topology, *, enter: Callable[[int, T | None], T], leave: Callable[[int, List[K]], K],
    root: int | np.integer = ..., mode: Literal["dfs"] = ...,
) -> K: ...
# fmt: on
def traverse(topology: Topology, *, mode="dfs", **kwargs):
    """Traverse nodes.

    Parameters
    ----------
    enter : (id: int, parent: T | None) => T, optional
        The callback when entering node, which accepts two parameters,
        the current node id and the return value of it parent node. In
        particular, the root node receives an `None`.
    leave : (id: int, children: List[T]) => T, optional
        The callback when leaving node. When leaving a node, subtree
        has already been traversed. Callback accepts two parameters,
        the current node id and list of the return value of children,
        In particular, the leaf node receives an empty list.
    root : int, default to `0`
        Start from the root node of the subtree
    mode : `dfs`, default to `dfs`
    """

    match mode:
        case "dfs":
            return _traverse_dfs(topology, **kwargs)
        case _:
            raise ValueError(f"unsupported mode: `{mode}`")


def _traverse_dfs(topology: Topology, *, enter=None, leave=None, root=0):
    """Traverse each nodes by dfs."""
    children_map = dict[int, list[int]]()
    for idx, pid in zip(*topology):
        children_map.setdefault(pid, [])
        children_map[pid].append(idx)

    # manual dfs to avoid stack overflow in long branch
    stack: List[Tuple[int, bool]] = [(root, True)]  # (idx, is_enter)
    params = {root: None}
    vals = {}

    while len(stack) != 0:
        idx, is_enter = stack.pop()
        if is_enter:  # enter
            pre = params.pop(idx)
            cur = enter(idx, pre) if enter is not None else None
            stack.append((idx, False))
            for child in children_map.get(idx, []):
                stack.append((child, True))
                params[child] = cur
        else:  # leave
            children = [vals.pop(i) for i in children_map.get(idx, [])]
            vals[idx] = leave(idx, children) if leave is not None else None

    return vals[root]
