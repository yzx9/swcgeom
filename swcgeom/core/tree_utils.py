"""Tree utils."""

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from .tree import Tree

__all__ = ["to_sub_tree"]


def to_sub_tree(
    tree: Tree, sub_id: npt.ArrayLike, sub_pid: npt.ArrayLike
) -> Tuple[Tree, Dict[int, int]]:
    """Create sub tree from origin tree."""

    sub_id = np.array(sub_id, dtype=np.int32)
    sub_pid = np.array(sub_pid, dtype=np.int32)
    n_nodes = sub_id.shape[0]

    id_map = {idx: i for i, idx in enumerate(sub_id)}
    new_pid = [id_map[i] if i != -1 else -1 for i in sub_pid]

    ndata = {k: tree.get_ndata(k)[sub_id] for k in tree.keys()}
    ndata.update(
        id=np.arange(0, n_nodes),
        pid=np.array(new_pid, dtype=np.int32),
    )

    tree = Tree(n_nodes, **ndata)
    tree.source = tree.source
    return tree, id_map
