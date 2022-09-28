"""Tree utils."""

from typing import Callable, Dict, List, Tuple, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt

from .swc import SWCLike
from .tree import Tree

__all__ = ["REMOVE", "to_sub_tree", "cut_tree", "propagate_remove"]

REMOVE = -2
T, K = TypeVar("T"), TypeVar("K")


def to_sub_tree(
    swc_like: SWCLike, sub_id: npt.ArrayLike, sub_pid: npt.ArrayLike
) -> Tuple[Tree, Dict[int, int]]:
    """Create sub tree from origin tree.

    You can directly mark the node for removal, and we will
    automatically remove it, but if the node you remove is not a leaf
    node, you need to use `propagate_remove` to remove all children.
    """

    sub_id = np.array(sub_id, dtype=np.int32)
    sub_pid = np.array(sub_pid, dtype=np.int32)

    # remove nodes
    keeped_id = cast(npt.NDArray[np.bool_], sub_id != REMOVE)
    sub_id, sub_pid = sub_id[keeped_id], sub_pid[keeped_id]

    n_nodes = sub_id.shape[0]

    id_map = {idx: i for i, idx in enumerate(sub_id)}
    new_pid = [id_map[i] if i != -1 else -1 for i in sub_pid]

    ndata = {k: swc_like.get_ndata(k)[sub_id] for k in swc_like.keys()}
    ndata.update(
        id=np.arange(0, n_nodes),
        pid=np.array(new_pid, dtype=np.int32),
    )

    swc_like = Tree(n_nodes, **ndata)
    swc_like.source = swc_like.source
    return swc_like, id_map


@overload
def cut_tree(
    tree: Tree, *, enter: Callable[[Tree.Node, T | None], Tuple[T, bool]]
) -> Tree:
    ...


@overload
def cut_tree(
    tree: Tree, *, leave: Callable[[Tree.Node, list[K]], Tuple[K, bool]]
) -> Tree:
    ...


def cut_tree(
    tree: Tree,
    *,
    enter: Callable[[Tree.Node, T | None], Tuple[T, bool]] | None = None,
    leave: Callable[[Tree.Node, list[K]], Tuple[K, bool]] | None = None
):
    """Cut tree.

    Returning a `True` can delete the current node and all its
    children.
    """

    idx, pid = tree.id().copy(), tree.pid().copy()
    if enter:

        def _enter(n: Tree.Node, parent: Tuple[T, bool] | None) -> Tuple[T, bool]:
            if parent is not None and not parent[1]:
                return parent

            res, remove = enter(n, parent[0] if parent else None)
            if remove:
                idx[n.id] = REMOVE

            return res, True

        tree.traverse(enter=_enter)

    elif leave:

        def _leave(n: Tree.Node, children: List[K]) -> K:
            res, remove = leave(n, children)
            if remove:
                idx[n.id] = REMOVE

            return res

        tree.traverse(leave=_leave)
        propagate_remove(tree, idx)

    else:
        return tree.copy()

    new_tree, _ = to_sub_tree(tree, idx, pid)
    return new_tree


def propagate_remove(tree: Tree, idx: npt.NDArray[np.int32]) -> None:
    """Remove all children when parent is marked as removed."""

    def propagate(n: Tree.Node, remove_parent: bool | None) -> bool:
        remove = bool(remove_parent) or (idx[n.id] == REMOVE)
        if remove:
            idx[n.id] = REMOVE

        return remove

    tree.traverse(enter=propagate)
