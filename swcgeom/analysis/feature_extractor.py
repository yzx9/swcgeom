"""A seires of common feature."""

from functools import cached_property
from typing import Any, Dict, List, Literal, cast, overload

import numpy as np
import numpy.typing as npt

from ..core import Tree
from .branch import BranchAnalysis
from .node import NodeAnalysis
from .path import PathAnalysis
from .sholl import Sholl

__all__ = ["FeatureExtractor"]

Feature = Literal[
    "length",
    # node
    "node_radial_distance",
    "node_radial_distance_distribution",
    "node_branch_order",
    "node_branch_order_distribution",
    # branch
    "branch_length",
    "branch_length_distribution",
    "branch_tortuosity",
    "branch_tortuosity_distribution",
    # path
    "path_length",
    "path_length_distribution",
    "path_tortuosity",
    "path_tortuosity_distribution",
    # sholl
    "sholl",
]


class FeatureExtractor:
    """Extract feature from tree."""

    tree: Tree

    def __init__(self, tree: Tree) -> None:
        # TODO: support population
        self.tree = tree

    @overload
    def get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]:
        ...

    @overload
    def get(self, feature: List[Feature], **kwargs) -> List[npt.NDArray[np.float32]]:
        ...

    @overload
    def get(
        self, feature: Dict[Feature, Dict[str, Any]], **kwargs
    ) -> Dict[str, npt.NDArray[np.float32]]:
        ...

    def get(self, feature, **kwargs):
        """Get feature."""
        if isinstance(feature, dict):
            return {k: self._get(k, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self._get(k) for k in feature]

        return self._get(feature, **kwargs)

    def _get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]:
        get_feature = getattr(self, f"get_{feature}", None)
        if not callable(get_feature):
            raise ValueError(f"Invalid feature: {feature}")

        res = cast(npt.NDArray, get_feature(**kwargs))
        if res.dtype != np.float32:
            res = res.astype(np.float32)

        return res

    # Features

    def get_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return np.array([self.tree.length(**kwargs)], dtype=np.float32)

    # Node

    @cached_property
    def _node_analysis(self) -> NodeAnalysis:
        return NodeAnalysis(self.tree)

    def get_node_radial_distance(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._node_analysis.get_radial_distance(**kwargs)

    def get_node_radial_distance_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._node_analysis.get_radial_distance_distribution(**kwargs)

    def get_node_branch_order(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._node_analysis.get_branch_order(**kwargs)

    def get_node_branch_order_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._node_analysis.get_branch_order_distribution(**kwargs)

    # Branch

    @cached_property
    def _branch_anlysis(self) -> BranchAnalysis:
        return BranchAnalysis(self.tree)

    def get_branch_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._branch_anlysis.get_length(**kwargs)

    def get_branch_length_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._branch_anlysis.get_length_distribution(**kwargs)

    def get_branch_tortuosity(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._branch_anlysis.get_tortuosity(**kwargs)

    def get_branch_tortuosity_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._branch_anlysis.get_tortuosity_distribution(**kwargs)

    # Path

    @cached_property
    def _path_analysis(self) -> PathAnalysis:
        return PathAnalysis(self.tree)

    def get_path_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._path_analysis.get_length(**kwargs)

    def get_path_length_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._path_analysis.get_length_distribution(**kwargs)

    def get_path_tortuosity(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._path_analysis.get_tortuosity(**kwargs)

    def get_path_tortuosity_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._path_analysis.get_tortuosity_distribution(**kwargs)

    # Sholl

    def get_sholl(self, **kwargs) -> npt.NDArray[np.int32]:
        return Sholl(self.tree, **kwargs).get_count()
