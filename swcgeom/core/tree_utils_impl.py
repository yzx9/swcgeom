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


"""SWC util wrapper for tree, split to avoid circle imports.

Notes
-----
Do not import `Tree` and keep this file minimized.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from swcgeom.core.swc import SWCLike, SWCNames
from swcgeom.core.swc_utils import Topology, to_sub_topology, traverse

__all__ = ["get_subtree_impl", "to_subtree_impl"]

Mapping = dict[int, int] | list[int]
TreeArgs = tuple[int, dict[str, npt.NDArray[Any]], str, SWCNames]


def get_subtree_impl(
    swc_like: SWCLike, n: int, *, out_mapping: Optional[Mapping] = None
) -> TreeArgs:
    ids = []
    topo = (swc_like.id(), swc_like.pid())
    traverse(topo, enter=lambda n, _: ids.append(n), root=n)

    sub_ids = np.array(ids, dtype=np.int32)
    sub_pid = swc_like.pid()[sub_ids]
    sub_pid[0] = -1
    return to_subtree_impl(swc_like, (sub_ids, sub_pid), out_mapping=out_mapping)


def to_subtree_impl(
    swc_like: SWCLike,
    sub: Topology,
    *,
    out_mapping: Optional[Mapping] = None,
) -> TreeArgs:
    (new_id, new_pid), mapping = to_sub_topology(sub)

    n_nodes = new_id.shape[0]
    ndata = {k: swc_like.get_ndata(k)[mapping].copy() for k in swc_like.keys()}
    ndata.update(id=new_id, pid=new_pid)

    if isinstance(out_mapping, list):
        out_mapping.clear()
        out_mapping.extend(mapping)
    elif isinstance(out_mapping, dict):
        out_mapping.clear()
        for new_id, old_id in enumerate(mapping):
            out_mapping[new_id] = old_id  # returning a dict may leads to bad perf

    return n_nodes, ndata, swc_like.source, swc_like.names
