# Copyright 2022-2025 Zexin Yuan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Base SWC format utils."""

from collections.abc import Callable
from typing import Literal, NamedTuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "Topology",
    "SWCNames",
    "swc_names",  # may not need to export
    "get_names",
    "SWCTypes",
    "get_types",
    "get_topology",
    "get_dsu",
    "traverse",
]

T = TypeVar("T")
K = TypeVar("K")
Topology = tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]  # (id, pid)


class SWCNames(NamedTuple):
    """SWC format column names."""

    id: str = "id"
    type: str = "type"
    x: str = "x"
    y: str = "y"
    z: str = "z"
    r: str = "r"
    pid: str = "pid"

    def cols(self) -> list[str]:
        return [self.id, self.type, self.x, self.y, self.z, self.r, self.pid]


swc_names = SWCNames()


def get_names(names: SWCNames | None = None) -> SWCNames:
    return names or swc_names


class SWCTypes(NamedTuple):
    """SWC format types.

    See Also
    ---------
    NeuroMoprho.org - What is SWC format?
        https://neuromorpho.org/myfaq.jsp
    """

    undefined: int = 0
    soma: int = 1
    axon: int = 2
    basal_dendrite: int = 3
    apical_dendrite: int = 4
    custom: int = 5  # user-defined preferences
    unspecified_neurites: int = 6
    glia_processes: int = 7


swc_types = SWCTypes()


def get_types(types: SWCTypes | None = None) -> SWCTypes:
    return types or swc_types


def get_topology(df: pd.DataFrame, *, names: SWCNames | None = None) -> Topology:
    names = get_names(names)
    return (df[names.id].to_numpy(), df[names.pid].to_numpy())


def get_dsu(
    df: pd.DataFrame, *, names: SWCNames | None = None
) -> npt.NDArray[np.int32]:
    """Get disjoint set union."""
    names = get_names(names)
    dsu = np.where(
        df[names.pid] == -1, df[names.id], df[names.pid]
    )  # Disjoint Set Union

    id2idx = dict(zip(df[names.id], range(len(df))))
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


@overload
def traverse(
    topology: Topology,
    *,
    enter: Callable[[int, T | None], T],
    root: int | np.integer = ...,
    mode: Literal["dfs"] = ...,
) -> None: ...
@overload
def traverse(
    topology: Topology,
    *,
    leave: Callable[[int, list[K]], K],
    root: int | np.integer = ...,
    mode: Literal["dfs"] = ...,
) -> K: ...
@overload
def traverse(
    topology: Topology,
    *,
    enter: Callable[[int, T | None], T],
    leave: Callable[[int, list[K]], K],
    root: int | np.integer = ...,
    mode: Literal["dfs"] = ...,
) -> K: ...
def traverse(topology: Topology, *, mode="dfs", **kwargs):
    """Traverse nodes.

    Parameters
    ----------
    enter : (id: int, parent: T | None) => T, optional
        The callback when entering node, which accepts two parameters,
        the current node id and the return value of it parent node. In
        particular, the root node receives an `None`.
    leave : (id: int, children: list[T]) => T, optional
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
    stack: list[tuple[int, bool]] = [(root, True)]  # (idx, is_enter)
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
