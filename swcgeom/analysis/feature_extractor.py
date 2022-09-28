"""A seires of common feature.

Notes
-----
For development, see method `FeatureExtractor._gen_feature_calculator`
to confirm the naming specification.
"""

from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, overload

import numpy as np
import numpy.typing as npt

from ..core import Tree
from .branch_features import BranchFeatures
from .node_features import NodeFeatures
from .path_features import PathFeatures
from .sholl import Sholl

__all__ = ["FeatureExtractor"]

Feature = Literal[
    "length",
    "sholl",
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
            return {k: self._get_feature(k, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self._get_feature(k) for k in feature]

        return self._get_feature(feature, **kwargs)

    def _get_feature(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]:
        calc = self._gen_feature_calculator(feature)
        return self._calc_feature(calc, **kwargs)

    def _gen_feature_calculator(self, feature: Feature) -> Callable[[], npt.NDArray]:
        if callable(calc := getattr(self, f"get_{feature}", None)):
            return calc  # custom features

        components = feature.split("_")
        attr = f"_{components[0]}_features"
        method = f"get_{'_'.join(components[1:])}"
        if (module := getattr(self, attr, None)) and callable(
            calc := getattr(module, method, None)
        ):
            return calc

        raise ValueError(f"Invalid feature: {feature}")

    def _calc_feature(
        self, calculator: Callable[[], npt.NDArray], **kwargs
    ) -> npt.NDArray[np.float32]:
        res = calculator(**kwargs)
        if res.dtype != np.float32:
            res = res.astype(np.float32)

        return res

    # Features

    def get_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return np.array([self.tree.length(**kwargs)], dtype=np.float32)

    def get_sholl(self, **kwargs) -> npt.NDArray[np.int32]:
        return Sholl(self.tree, **kwargs).get_count()

    # Modules

    @cached_property
    def _node_features(self) -> NodeFeatures:
        return NodeFeatures(self.tree)

    @cached_property
    def _branch_features(self) -> BranchFeatures:
        return BranchFeatures(self.tree)

    @cached_property
    def _path_features(self) -> PathFeatures:
        return PathFeatures(self.tree)
