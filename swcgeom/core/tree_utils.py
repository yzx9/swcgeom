"""SWC util wrapper for tree."""

import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, overload

import numpy as np

from .swc import SWCLike
from .swc_utils import (
    REMOVAL,
    SWCNames,
    Topology,
    get_names,
    is_bifurcate,
    propagate_removal,
    sort_nodes_impl,
    to_sub_topology,
    traverse,
)
from .tree import Tree

__all__ = [
    "sort_tree",
    "cut_tree",
    "to_sub_tree",
    "to_subtree",
    "get_subtree",
    "redirect_tree",
    "cat_tree",
]

T, K = TypeVar("T"), TypeVar("K")
EPS = 1e-5


def is_binary_tree(tree: Tree, exclude_soma: bool = True) -> bool:
    """Check is it a bifurcate tree."""
    return is_bifurcate((tree.id(), tree.pid()), exclude_root=exclude_soma)


def sort_tree(tree: Tree) -> Tree:
    """Sort the indices of neuron tree.

    See Also
    --------
    ~.core.swc_utils.sort_nodes
    """
    return _sort_tree(tree.copy())


# fmt:off
@overload
def cut_tree(tree: Tree, *, enter: Callable[[Tree.Node, T | None], Tuple[T, bool]]) -> Tree: ...
@overload
def cut_tree(tree: Tree, *, leave: Callable[[Tree.Node, List[K]], Tuple[K, bool]]) -> Tree: ...
# fmt:on
def cut_tree(tree: Tree, *, enter=None, leave=None):
    """Traverse and cut the tree.

    Returning a `True` can delete the current node and its children.
    """

    removals: List[int] = []

    if enter:

        def _enter(n: Tree.Node, parent: Tuple[T, bool] | None) -> Tuple[T, bool]:
            if parent is not None and parent[1]:
                removals.append(n.id)
                return parent

            res, removal = enter(n, parent[0] if parent else None)
            if removal:
                removals.append(n.id)

            return res, removal

        tree.traverse(enter=_enter)

    elif leave:

        def _leave(n: Tree.Node, children: List[K]) -> K:
            res, removal = leave(n, children)
            if removal:
                removals.append(n.id)

            return res

        tree.traverse(leave=_leave)

    else:
        return tree.copy()

    return to_subtree(tree, removals)


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
    warnings.warn(
        "`to_sub_tree` will be removed in v0.6.0, it is replaced by "
        "`to_subtree` beacuse it is easy to use, and this will be "
        "removed in next version",
        DeprecationWarning,
    )

    sub = propagate_removal(sub)
    (new_id, new_pid), id_map_arr = to_sub_topology(sub)

    n_nodes = new_id.shape[0]
    ndata = {k: swc_like.get_ndata(k)[id_map_arr].copy() for k in swc_like.keys()}
    ndata.update(id=new_id, pid=new_pid)

    subtree = Tree(n_nodes, **ndata)
    subtree.source = swc_like.source

    id_map = {}
    for i, idx in enumerate(id_map_arr):
        id_map[idx] = i
    return subtree, id_map


def to_subtree(swc_like: SWCLike, removals: Iterable[int]) -> Tree:
    """Create subtree from origin tree.

    Parameters
    ----------
    swc_like : SWCLike
    removals : List of int
        A list of id of nodes to be removed.
    """
    new_ids = swc_like.id().copy()
    for i in removals:
        new_ids[i] = REMOVAL

    sub = propagate_removal((new_ids, swc_like.pid()))
    return _to_subtree(swc_like, sub)


def get_subtree(swc_like: SWCLike, n: int) -> Tree:
    """Get subtree rooted at n.

    Parameters
    ----------
    swc_like : SWCLike
    n : int
        Id of the root of the subtree.
    """
    ids = []
    topo = (swc_like.id(), swc_like.pid())
    traverse(topo, enter=lambda n, _: ids.append(n), root=n)

    sub_ids = np.array(ids, dtype=np.int32)
    sub_pid = swc_like.pid()[sub_ids]
    sub_pid[0] = -1
    return _to_subtree(swc_like, (sub_ids, sub_pid))


def redirect_tree(tree: Tree, new_root: int, sort: bool = True) -> Tree:
    """Set root to new point and redirect tree graph.

    Parameter
    ---------
    tree : Tree
        The tree.
    new_root : int
        The id of new root.
    sort : bool, default `True`
        If true, sort indices of nodes after redirect.
    """
    tree = tree.copy()
    path = [tree.node(new_root)]
    while (p := path[-1]).parent() is not None:
        path.append(p)

    for n, p in zip(path[1:], path[:-1]):
        n.pid = p.id

    if sort:
        _sort_tree(tree)

    return tree


def cat_tree(  # pylint: disable=too-many-arguments
    tree1: Tree,
    tree2: Tree,
    node1: int,
    node2: int = 0,
    *,
    no_move: bool = False,
    names: Optional[SWCNames] = None,
) -> Tree:
    """Concatenates the second tree onto the first one.

    Paramters
    ---------
    tree1 : Tree
    tree2 : Tree
    node1 : int
        The node id of the tree to be connected.
    node2 : int, default `0`
        The node id of the connection point.
    no_move : bool, default `False`
        If true, link connection point without move.
    """
    names = get_names(names)
    tree, tree2 = tree1.copy(), tree2.copy()
    if not tree2.node(node2).is_soma():
        tree2 = redirect_tree(tree2, node2, sort=False)

    c = tree.node(node1)
    if not no_move:
        tree2.ndata[names.x] -= tree2.node(node2).x - c.x
        tree2.ndata[names.y] -= tree2.node(node2).y - c.y
        tree2.ndata[names.z] -= tree2.node(node2).z - c.z

    tree2.ndata[names.id] += tree.number_of_nodes()
    tree2.ndata[names.pid] += tree.number_of_nodes()
    if np.linalg.norm(tree2.node(node2).xyz() - c.xyz()) < EPS:
        for n in tree2.node(node2).children():
            n.pid = node1
    else:
        tree2.node(node2).pid = node1

    for k, v in tree.ndata.items():  # only keep keys in tree1
        if k in tree2.ndata:
            tree.ndata[k] = np.concatenate([v, tree2.ndata[k]])
        else:
            tree.ndata[k] = np.pad(v, (0, tree2.number_of_nodes()))

    return _sort_tree(tree)


def _sort_tree(tree: Tree) -> Tree:
    """Sort the indices of neuron tree inplace."""
    (new_ids, new_pids), id_map = sort_nodes_impl((tree.id(), tree.pid()))
    tree.ndata = {k: tree.ndata[k][id_map] for k in tree.ndata}
    tree.ndata.update(id=new_ids, pid=new_pids)
    return tree


def _to_subtree(swc_like: SWCLike, sub: Topology) -> Tree:
    (new_id, new_pid), id_map = to_sub_topology(sub)

    n_nodes = new_id.shape[0]
    ndata = {k: swc_like.get_ndata(k)[id_map].copy() for k in swc_like.keys()}
    ndata.update(id=new_id, pid=new_pid)

    subtree = Tree(n_nodes, **ndata)
    subtree.source = swc_like.source
    return subtree
