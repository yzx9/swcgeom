
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import numpy as np
from typing_extensions import override

from swcgeom.core import Branch, BranchTree, Node, Tree
from swcgeom.transforms.base import Transform

__all__ = ["BranchTreeAssembler"]


class BranchTreeAssembler(Transform[BranchTree, Tree]):
    EPS = 1e-6

    @override
    def __call__(self, x: BranchTree) -> Tree:
        nodes = [x.soma().detach()]
        stack = [(x.soma(), 0)]  # n_orig, id_new
        while len(stack):
            n_orig, pid_new = stack.pop()
            children = n_orig.children()

            for br, c in self.pair(x.branches.get(n_orig.id, []), children):
                s = 1 if np.linalg.norm(br[0].xyz() - n_orig.xyz()) < self.EPS else 0
                e = -2 if np.linalg.norm(br[-1].xyz() - c.xyz()) < self.EPS else -1

                br_nodes = [n.detach() for n in br[s:e]] + [c.detach()]
                for i, n in enumerate(br_nodes):
                    # reindex
                    n.id = len(nodes) + i
                    n.pid = len(nodes) + i - 1

                br_nodes[0].pid = pid_new
                nodes.extend(br_nodes)
                stack.append((c, br_nodes[-1].id))

        return Tree(
            len(nodes),
            source=x.source,
            comments=x.comments,
            names=x.names,
            **{
                k: np.array([n.__getattribute__(k) for n in nodes])
                for k in x.names.cols()
            },
        )

    def pair(
        self, branches: list[Branch], endpoints: list[Node]
    ) -> Iterable[tuple[Branch, Node]]:
        assert len(branches) == len(endpoints)
        xyz1 = [br[-1].xyz() for br in branches]
        xyz2 = [n.xyz() for n in endpoints]
        v = np.reshape(xyz1, (-1, 1, 3)) - np.reshape(xyz2, (1, -1, 3))
        dis = np.linalg.norm(v, axis=-1)

        # greedy algorithm
        pairs = []
        for _ in range(len(branches)):
            # find minimal
            min_idx = np.argmin(dis)
            min_branch_idx, min_endpoint_idx = np.unravel_index(min_idx, dis.shape)
            pairs.append((branches[min_branch_idx], endpoints[min_endpoint_idx]))

            # remove current node
            dis[min_branch_idx, :] = np.inf
            dis[:, min_endpoint_idx] = np.inf

        return pairs
