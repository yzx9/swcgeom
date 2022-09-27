"""A seires of common feature."""

from functools import cached_property
from typing import Any, Dict, List, Literal, overload

import numpy as np
import numpy.typing as npt

from ..core import Tree
from .branch import BranchAnalysis
from .depth import DepthAnalysis
from .path import PathAnalysis
from .sholl import Sholl

__all__ = ["FeatureExtractor"]

Feature = Literal[
    "length",
    "branch_length",
    "branch_length_distribution",
    "branch_depth",
    "tip_depth",
    "path_length",
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
            raise ValueError(f"Invalid feture: {feature}")

        return get_feature(**kwargs)

    # Features

    def get_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return np.array([self.tree.length(**kwargs)], dtype=np.float32)

    def get_branch_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._branch_anlysis.get_length(**kwargs)

    def get_branch_length_distribution(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._branch_anlysis.get_length_distribution(**kwargs).astype(np.float32)

    def get_branch_depth(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._depth_analysis.get_branch_depth(**kwargs).astype(np.float32)

    def get_tip_depth(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._depth_analysis.get_tip_depth(**kwargs).astype(np.float32)

    def get_path_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return self._path_analysis.get_length(**kwargs)

    def get_sholl(self, **kwargs) -> npt.NDArray[np.float32]:
        return Sholl(self.tree, **kwargs).get_count().astype(np.float32)

    # Caches

    @cached_property
    def _branch_anlysis(self) -> BranchAnalysis:
        return BranchAnalysis(self.tree)

    @cached_property
    def _depth_analysis(self) -> DepthAnalysis:
        return DepthAnalysis(self.tree)

    @cached_property
    def _path_analysis(self) -> PathAnalysis:
        return PathAnalysis(self.tree)
