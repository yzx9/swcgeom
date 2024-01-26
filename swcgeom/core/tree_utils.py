"""SWC util wrapper for tree."""

import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, overload

import numpy as np

from swcgeom.core.swc import SWCLike
from swcgeom.core.swc_utils import (
    REMOVAL,
    SWCNames,
    Topology,
    get_names,
    is_bifurcate,
    propagate_removal,
    sort_nodes_impl,
    to_sub_topology,
)
from swcgeom.core.tree import Tree
from swcgeom.core.tree_utils_impl import Mapping, get_subtree_impl, to_subtree_impl

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

    subtree = Tree(n_nodes, **ndata, source=swc_like.source, names=swc_like.names)

    id_map = {}
    for i, idx in enumerate(id_map_arr):
        id_map[idx] = i
    return subtree, id_map


def to_subtree(
    swc_like: SWCLike,
    removals: Iterable[int],
    *,
    out_mapping: Optional[Mapping] = None,
) -> Tree:
    """Create subtree from origin tree.

    Parameters
    ----------
    swc_like : SWCLike
    removals : List of int
        A list of id of nodes to be removed.
    out_mapping: List of int or Dict[int, int], optional
        Map new id to old id.
    """

    new_ids = swc_like.id().copy()
    for i in removals:
        new_ids[i] = REMOVAL

    sub = propagate_removal((new_ids, swc_like.pid()))
    n_nodes, ndata, source, names = to_subtree_impl(
        swc_like, sub, out_mapping=out_mapping
    )
    return Tree(n_nodes, **ndata, source=source, names=names)


def get_subtree(
    swc_like: SWCLike, n: int, *, out_mapping: Optional[Mapping] = None
) -> Tree:
    """Get subtree rooted at n.

    Parameters
    ----------
    swc_like : SWCLike
    n : int
        Id of the root of the subtree.
    out_mapping: List of int or Dict[int, int], optional
        Map new id to old id.
    """

    n_nodes, ndata, source, names = get_subtree_impl(
        swc_like, n, out_mapping=out_mapping
    )
    return Tree(n_nodes, **ndata, source=source, names=names)


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
    while (p := path[-1].parent()) is not None:
        path.append(p)

    path[0].pid = -1
    path[0].type, path[-1].type = path[-1].type, path[0].type
    for n, p in zip(path[1:], path[:-1]):
        n.pid = p.id

    if sort:
        _sort_tree(tree)

    return tree


def cat_tree(  # pylint: disable=too-many-arguments
    tree1: Tree,
    tree2: Tree,
    node1: int = 0,
    node2: int = 0,
    *,
    translate: bool = True,
    names: Optional[SWCNames] = None,
    no_move: Optional[bool] = None,  # legacy
) -> Tree:
    """Concatenates the second tree onto the first one.

    Paramters
    ---------
    tree1 : Tree
    tree2 : Tree
    node1 : int, default `0`
        The node id of the tree to be connected.
    node2 : int, default `0`
        The node id of the connection point.
    translate : bool, default `True`
        Wheather to translate node_2 to node_1. If False, add link
        between node_1 and node_2 without translate.
    """
    if no_move is not None:
        warnings.warn(
            "`no_move` has been, it is replaced by `translate` in "
            "v0.12.0, and this will be removed in next version",
            DeprecationWarning,
        )
        translate = not no_move

    names = get_names(names)
    tree, tree2 = tree1.copy(), tree2.copy()
    if not tree2.node(node2).is_root():
        tree2 = redirect_tree(tree2, node2, sort=False)

    c = tree.node(node1)
    if translate:
        tree2.ndata[names.x] -= tree2.node(node2).x - c.x
        tree2.ndata[names.y] -= tree2.node(node2).y - c.y
        tree2.ndata[names.z] -= tree2.node(node2).z - c.z

    ns = tree.number_of_nodes()
    if np.linalg.norm(tree2.node(node2).xyz() - c.xyz()) < EPS:
        remove = [node2 + ns]
        link_to_root = [n.id + ns for n in tree2.node(node2).children()]
    else:
        remove = None
        link_to_root = [node2 + ns]

    # APIs of tree2 are no longer available since we modify the topology
    tree2.ndata[names.id] += ns
    tree2.ndata[names.pid] += ns

    for k, v in tree.ndata.items():  # only keep keys in tree1
        if k in tree2.ndata:
            tree.ndata[k] = np.concatenate([v, tree2.ndata[k]])
        else:
            tree.ndata[k] = np.pad(v, (0, tree2.number_of_nodes()))

    for n in link_to_root:
        tree.node(n).pid = node1

    if remove is not None:  # TODO: This should be easy to implement during sort
        for k, v in tree.ndata.items():
            tree.ndata[k] = np.delete(v, remove)

    _sort_tree(tree)
    return tree


def _sort_tree(tree: Tree) -> Tree:
    """Sort the indices of neuron tree inplace."""
    (new_ids, new_pids), id_map = sort_nodes_impl((tree.id(), tree.pid()))
    tree.ndata = {k: tree.ndata[k][id_map] for k in tree.ndata}
    tree.ndata.update(id=new_ids, pid=new_pids)
    return tree
