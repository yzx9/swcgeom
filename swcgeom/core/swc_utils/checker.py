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


"""Check common"""

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import deprecated

from swcgeom.core.swc_utils.base import SWCNames, Topology, get_dsu, get_names, traverse
from swcgeom.utils import DisjointSetUnion

__all__ = [
    "is_single_root",
    "is_bifurcate",
    "is_sorted",
    "has_cyclic",
    # legacy
    "is_binary_tree",
    "check_single_root",
]


def is_single_root(df: pd.DataFrame, *, names: Optional[SWCNames] = None) -> bool:
    """Check is it only one root."""
    return len(np.unique(get_dsu(df, names=names))) == 1


def is_bifurcate(topology: Topology, *, exclude_root: bool = True) -> bool:
    """Check is it a bifurcate topology."""

    children = defaultdict(list)
    for idx, pid in zip(*topology):
        children[pid].append(idx)

    root = children[-1]
    for k, v in children.items():
        if len(v) > 1 and (not exclude_root or k in root):
            return False

    return True


def is_sorted(topology: Topology) -> bool:
    """Check is it sorted.

    In a sorted topology, parent samples should appear before any child
    samples.
    """
    flag = True

    def enter(idx: int, parent: int | None) -> int:
        nonlocal flag
        if parent is not None and idx < parent:
            flag = False

        return idx

    traverse(topology=topology, enter=enter)
    return flag


def has_cyclic(topology: Topology) -> bool:
    """Has cyclic in topology."""
    node_num = len(topology[0])
    dsu = DisjointSetUnion(node_number=node_num)

    for i in range(node_num):
        node_a = topology[0][i]
        node_b = topology[1][i]
        # skip the root node
        if node_b == -1:
            continue

        # check whether it is circle
        if dsu.is_same_set(node_a, node_b):
            return True

        dsu.union_sets(node_a, node_b)

    return False


@deprecated("Use `is_single_root` instead")
def check_single_root(*args, **kwargs) -> bool:
    """Check if the tree is single root.

    .. deprecated:: 0.5.0
        Use :meth:`is_single_root` instead.
    """

    return is_single_root(*args, **kwargs)


@deprecated("Use `is_bifurcate` instead")
def is_binary_tree(
    df: pd.DataFrame, exclude_root: bool = True, *, names: Optional[SWCNames] = None
) -> bool:
    """Check is it a binary tree.

    .. deprecated:: 0.8.0
        Use :meth:`is_bifurcate` instead.
    """

    names = get_names(names)
    topo = (df[names.id].to_numpy(), df[names.pid].to_numpy())
    return is_bifurcate(topo, exclude_root=exclude_root)
