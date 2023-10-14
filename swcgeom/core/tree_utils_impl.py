"""SWC util wrapper for tree, split to avoid circle imports.

Notes
-----
Do not import `Tree` and keep this file minimized.
"""

from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

from swcgeom.core.swc import SWCLike, SWCNames
from swcgeom.core.swc_utils import Topology, to_sub_topology, traverse

__all__ = ["get_subtree_impl", "to_subtree_impl"]

TreeArgs = Tuple[int, Dict[str, npt.NDArray[Any]], str, SWCNames]


def get_subtree_impl(swc_like: SWCLike, n: int) -> TreeArgs:
    ids = []
    topo = (swc_like.id(), swc_like.pid())
    traverse(topo, enter=lambda n, _: ids.append(n), root=n)

    sub_ids = np.array(ids, dtype=np.int32)
    sub_pid = swc_like.pid()[sub_ids]
    sub_pid[0] = -1
    return to_subtree_impl(swc_like, (sub_ids, sub_pid))


def to_subtree_impl(swc_like: SWCLike, sub: Topology) -> TreeArgs:
    (new_id, new_pid), id_map = to_sub_topology(sub)

    n_nodes = new_id.shape[0]
    ndata = {k: swc_like.get_ndata(k)[id_map].copy() for k in swc_like.keys()}
    ndata.update(id=new_id, pid=new_pid)

    return n_nodes, ndata, swc_like.source, swc_like.names
