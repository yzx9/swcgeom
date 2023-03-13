"""Cut subtree.

This module provides a series of low-level topological subtree methods,
but in more cases, you can use the high-level methods provided in
`tree_utils`, which wrap the methods in this module and provide a
high-level API.
"""

from typing import Tuple, cast

import numpy as np
import numpy.typing as npt

from .base import Topology, traverse

__all__ = ["REMOVAL", "to_sub_topology", "propagate_removal"]

REMOVAL = -2  # A marker in utils, place in the ids to mark it removal


def to_sub_topology(sub: Topology) -> Tuple[Topology, npt.NDArray[np.int32]]:
    """Create sub tree from origin tree.

    Mark the node to be removed, then use this method to get a child
    structure.

    Returns
    -------
    sub_topology : Topology
    id_map : List of int
        Map from new id to original id.

    See Also
    --------
    propagate_removal :
        If the node you remove is not a leaf node, you need to use it
        to mark all child nodes.
    """

    sub_id = np.array(sub[0], dtype=np.int32)
    sub_pid = np.array(sub[1], dtype=np.int32)

    # remove nodes
    keeped_id = cast(npt.NDArray[np.bool_], sub_id != REMOVAL)
    sub_id, sub_pid = sub_id[keeped_id], sub_pid[keeped_id]

    old2new = {idx: i for i, idx in enumerate(sub_id)}  # old idx to new id
    new_id = np.arange(0, sub_id.shape[0], dtype=np.int32)
    new_pid = np.array([old2new[i] if i != -1 else -1 for i in sub_pid], dtype=np.int32)

    return (new_id, new_pid), sub_id


def propagate_removal(topology: Topology) -> Topology:
    """Mark all children when parent is marked as removed.

    Returns
    -------
    new_topology : Topology
    """

    new_ids, pids = topology
    ids = np.arange(0, pids.shape[0])

    def propagate(n: int, parent: bool | None) -> bool:
        if remove := bool(parent) or (new_ids[n] == REMOVAL):
            new_ids[n] = REMOVAL

        return remove

    traverse((ids, pids), enter=propagate)
    return (new_ids, pids.copy())
