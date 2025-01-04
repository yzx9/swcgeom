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


"""Branch tree is a simplified neuron tree."""

import itertools

import numpy as np
import pandas as pd

from swcgeom.core.branch import Branch
from swcgeom.core.swc_utils import to_sub_topology
from swcgeom.core.tree import Tree

__all__ = ["BranchTree"]


class BranchTree(Tree):
    """Branch tree keeps topology of tree.

    A branch tree that contains only soma, branch, and tip nodes.
    """

    branches: dict[int, list[Branch]]

    def get_origin_branches(self) -> list[Branch]:
        """Get branches of original tree."""
        return list(itertools.chain(*self.branches.values()))

    def get_origin_node_branches(self, idx: int) -> list[Branch]:
        """Get branches of node of original tree."""
        return self.branches[idx]

    @classmethod
    def from_tree(cls, tree: Tree) -> "BranchTree":
        """Generating a branch tree from tree."""

        branches = tree.get_branches()

        sub_id = np.array([0] + [br[-1].id for br in branches], dtype=np.int32)
        sub_pid = np.array([-1] + [br[0].id for br in branches], dtype=np.int32)

        (new_id, new_pid), id_map = to_sub_topology((sub_id, sub_pid))

        n_nodes = new_id.shape[0]
        ndata = {k: tree.get_ndata(k)[id_map].copy() for k in tree.keys()}
        ndata.update(id=new_id, pid=new_pid)

        branch_tree = cls(n_nodes, **ndata, source=tree.source, names=tree.names)

        branch_tree.branches = {}
        for br in branches:
            idx = np.nonzero(id_map == br[0].id)[0][0].item()
            branch_tree.branches.setdefault(idx, [])
            branch_tree.branches[idx].append(br.detach())

        return branch_tree

    @classmethod
    def from_data_frame(cls, df: pd.DataFrame, *args, **kwargs) -> "BranchTree":
        tree = super().from_data_frame(df, *args, **kwargs)
        return cls.from_tree(tree)
