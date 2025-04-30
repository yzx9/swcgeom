
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Transformation in tree."""

from collections.abc import Callable

import numpy as np
from typing_extensions import deprecated, override

from swcgeom.core import Branch, BranchTree, DictSWC, Path, Tree, cut_tree, to_subtree
from swcgeom.core.swc_utils import SWCTypes, get_types
from swcgeom.transforms.base import Transform
from swcgeom.transforms.branch import BranchConvSmoother, BranchIsometricResampler
from swcgeom.transforms.branch_tree import BranchTreeAssembler
from swcgeom.transforms.geometry import Normalizer

__all__ = [
    "ToBranchTree",
    "ToLongestPath",
    "TreeSmoother",
    "TreeNormalizer",
    "CutByType",
    "CutAxonTree",
    "CutDendriteTree",
    "CutByFurcationOrder",
    "CutShortTipBranch",
    "IsometricResampler",
]


# pylint: disable=too-few-public-methods
class ToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    @override
    def __call__(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)


class ToLongestPath(Transform[Tree, Path[DictSWC]]):
    """Transform tree to longest path."""

    def __init__(self, *, detach: bool = True) -> None:
        self.detach = detach

    @override
    def __call__(self, x: Tree) -> Path[DictSWC]:
        paths = x.get_paths()
        idx = np.argmax([p.length() for p in paths])
        path = paths[idx]
        if self.detach:
            path = path.detach()
        return path  # type: ignore


class TreeSmoother(Transform[Tree, Tree]):  # pylint: disable=missing-class-docstring
    def __init__(self, n_nodes: int = 5) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.trans = BranchConvSmoother(n_nodes=n_nodes)

    @override
    def __call__(self, x: Tree) -> Tree:
        x = x.copy()
        for br in x.get_branches():
            # TODO: works but is weird
            smoothed = self.trans(br)
            x.ndata["x"][br.origin_id()] = smoothed.x()
            x.ndata["y"][br.origin_id()] = smoothed.y()
            x.ndata["z"][br.origin_id()] = smoothed.z()

        return x

    @override
    def extra_repr(self) -> str:
        return f"n_nodes={self.n_nodes}"


@deprecated("Use `Normalizer` instead")
class TreeNormalizer(Normalizer[Tree]):
    """Noramlize coordinates and radius to 0-1.

    .. deprecated:: 0.6.0
        Use :cls:`Normalizer` instead.
    """


class CutByType(Transform[Tree, Tree]):
    """Cut tree by type.

    In order to preserve the tree structure, all ancestor nodes of the node to be preserved will be preserved.

    NOTE: Not all reserved nodes are of the specified type.
    """

    def __init__(self, type: int) -> None:  # pylint: disable=redefined-builtin
        super().__init__()
        self.type = type

    @override
    def __call__(self, x: Tree) -> Tree:
        removals = set(x.id()[x.type() != self.type])

        def leave(n: Tree.Node, keep_children: list[bool]) -> bool:
            if n.id in removals and any(keep_children):
                removals.remove(n.id)
            return n.id not in removals

        x.traverse(leave=leave)
        y = to_subtree(x, removals)
        return y

    @override
    def extra_repr(self) -> str:
        return f"type={self.type}"


class CutAxonTree(CutByType):
    """Cut axon tree."""

    def __init__(self, types: SWCTypes | None = None) -> None:
        types = get_types(types)
        super().__init__(type=types.axon)


class CutDendriteTree(CutByType):
    """Cut dendrite tree."""

    def __init__(self, types: SWCTypes | None = None) -> None:
        types = get_types(types)
        super().__init__(type=types.basal_dendrite)  # TODO: apical dendrite


class CutByFurcationOrder(Transform[Tree, Tree]):
    """Cut tree by furcation order."""

    max_furcation_order: int

    def __init__(self, max_bifurcation_order: int) -> None:
        self.max_furcation_order = max_bifurcation_order

    @override
    def __call__(self, x: Tree) -> Tree:
        return cut_tree(x, enter=self._enter)

    def __repr__(self) -> str:
        return f"CutByBifurcationOrder-{self.max_furcation_order}"

    def _enter(self, n: Tree.Node, parent_level: int | None) -> tuple[int, bool]:
        if parent_level is None:
            level = 0
        elif n.is_furcation():
            level = parent_level + 1
        else:
            level = parent_level
        return (level, level >= self.max_furcation_order)


@deprecated("Use CutByFurcationOrder instead")
class CutByBifurcationOrder(CutByFurcationOrder):
    """Cut tree by bifurcation order.

    NOTE: Deprecated due to the wrong spelling of furcation. For now, it
    is just an alias of `CutByFurcationOrder` and raise a warning. It
    will be change to raise an error in the future.
    """

    max_furcation_order: int

    def __init__(self, max_bifurcation_order: int) -> None:
        super().__init__(max_bifurcation_order)

    def __repr__(self) -> str:
        return f"CutByBifurcationOrder-{self.max_furcation_order}"


class CutShortTipBranch(Transform[Tree, Tree]):
    """Cut off too short terminal branches.

    This method is usually applied in the post-processing of manual
    reconstruction. When the user draw lines, a line head is often left
    at the junction of two lines.
    """

    thre: float
    callbacks: list[Callable[[Tree.Branch], None]]

    def __init__(
        self, thre: float = 5, callback: Callable[[Tree.Branch], None] | None = None
    ) -> None:
        self.thre = thre
        self.callbacks = []

        if callback is not None:
            self.callbacks.append(callback)

    @override
    def __call__(self, x: Tree) -> Tree:
        removals: list[int] = []
        self.callbacks.append(lambda br: removals.append(br[1].id))
        x.traverse(leave=self._leave)
        self.callbacks.pop()
        return to_subtree(x, removals)

    @override
    def extra_repr(self) -> str:
        return f"threshold={self.thre}"

    def _leave(
        self, n: Tree.Node, children: list[tuple[float, Tree.Node] | None]
    ) -> tuple[float, Tree.Node] | None:
        if len(children) == 0:  # tip
            return 0, n

        if len(children) == 1 and children[0] is not None:  # elongation
            dis, child = children[0]
            dis += n.distance(child)
            return dis, n

        for c in children:
            if c is None:
                continue

            dis, child = c
            if dis + n.distance(child) > self.thre:
                continue

            path = [n.id]  # n does not delete, but will include in callback
            while child is not None:  # TODO: perf
                path.append(child.id)
                child = cc[0] if len((cc := child.children())) > 0 else None

            br = Tree.Branch(n.attach, path)
            for cb in self.callbacks:
                cb(br)

        return None


class Resampler(Transform[Tree, Tree]):
    def __init__(self, branch_resampler: Transform[Branch, Branch]) -> None:
        super().__init__()
        self.resampler = branch_resampler
        self.assembler = BranchTreeAssembler()

    @override
    def __call__(self, x: Tree) -> Tree:
        t = BranchTree.from_tree(x)
        t.branches = {
            k: [self.resampler(br) for br in brs] for k, brs in t.branches.items()
        }
        return self.assembler(t)


class IsometricResampler(Resampler):
    def __init__(
        self, distance: float, *, adjust_last_gap: bool = True, **kwargs
    ) -> None:
        branch_resampler = BranchIsometricResampler(
            distance, adjust_last_gap=adjust_last_gap, **kwargs
        )
        super().__init__(branch_resampler)
