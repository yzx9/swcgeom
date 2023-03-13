"""SWC util wrapper for tree."""

from typing import Callable, Dict, List, Tuple, TypeVar, overload

from .swc import SWCLike
from .swc_utils import (
    REMOVAL,
    Topology,
    propagate_removal,
    sort_nodes_impl,
    to_sub_topology,
)
from .tree import Tree

__all__ = ["REMOVAL", "sort_tree", "to_sub_tree", "cut_tree"]

T, K = TypeVar("T"), TypeVar("K")


def sort_tree(tree: Tree) -> Tree:
    """Sort the indices of neuron tree.

    See Also
    --------
    ~swc_utils.sort_nodes
    """
    indices, new_ids, new_pids = sort_nodes_impl(tree.id(), tree.pid())
    new_tree = tree.copy()
    new_tree.ndata = {k: tree.ndata[k][indices] for k in tree.ndata}
    new_tree.ndata["id"] = new_ids
    new_tree.ndata["pid"] = new_pids
    return new_tree


def to_sub_tree(swc_like: SWCLike, sub: Topology) -> Tuple[Tree, Dict[int, int]]:
    """Create subtree from origin tree.

    You can directly mark the node for removal, and we will remove it,
    but if the node you remove is not a leaf node, you need to use
    `propagate_remove` to remove all children.

    Returns
    -------
    tree : Tree
    id_map : Dict[int, int]
    """
    (new_id, new_pid), id_map = to_sub_topology(sub)

    # TODO: perf
    sub_id = list(id_map.keys())

    n_nodes = new_id.shape[0]
    ndata = {k: swc_like.get_ndata(k)[sub_id].copy() for k in swc_like.keys()}
    ndata.update(id=new_id, pid=new_pid)

    new_tree = Tree(n_nodes, **ndata)
    new_tree.source = swc_like.source
    return new_tree, id_map


# fmt:off
@overload
def cut_tree(tree: Tree, *, enter: Callable[[Tree.Node, T | None], Tuple[T, bool]]) -> Tree: ...
@overload
def cut_tree(tree: Tree, *, leave: Callable[[Tree.Node, list[K]], Tuple[K, bool]]) -> Tree: ...
# fmt:on


def cut_tree(tree: Tree, *, enter=None, leave=None):
    """Traverse and cut the tree.

    Returning a `True` can delete the current node and its children.
    """

    if enter:
        new_ids = tree.id().copy()

        def _enter(n: Tree.Node, parent: Tuple[T, bool] | None) -> Tuple[T, bool]:
            if parent is not None and parent[1]:
                new_ids[n.id] = REMOVAL
                return parent

            res, remove = enter(n, parent[0] if parent else None)
            if remove:
                new_ids[n.id] = REMOVAL

            return res, remove

        tree.traverse(enter=_enter)
        sub = (new_ids, tree.pid().copy())

    elif leave:
        new_ids = tree.id().copy()

        def _leave(n: Tree.Node, children: List[K]) -> K:
            res, remove = leave(n, children)
            if remove:
                new_ids[n.id] = REMOVAL

            return res

        tree.traverse(leave=_leave)
        sub = propagate_removal((new_ids, tree.pid()))

    else:
        return tree.copy()

    new_tree, _ = to_sub_tree(tree, sub)
    return new_tree
