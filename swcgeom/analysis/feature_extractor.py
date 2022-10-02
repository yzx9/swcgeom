"""A seires of common feature.

Notes
-----
For development, see method `FeatureExtractor._gen_feature_calculator`
to confirm the naming specification.
"""

from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, cast, overload

import numpy as np
import numpy.typing as npt

from ..core import Tree
from ..utils import to_distribution
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
    "node_branch_order",
    # branch
    "branch_length",
    "branch_tortuosity",
    # path
    "path_length",
    "path_tortuosity",
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

    @overload
    def get_distribution(
        self, feature: Feature, step: float = ..., **kwargs
    ) -> npt.NDArray[np.int32]:
        ...

    @overload
    def get_distribution(
        self, feature: List[Feature], step: float = ..., **kwargs
    ) -> List[npt.NDArray[np.int32]]:
        ...

    @overload
    def get_distribution(
        self, feature: Dict[Feature, Dict[str, Any]], step: float = ..., **kwargs
    ) -> Dict[str, npt.NDArray[np.int32]]:
        ...

    def get_distribution(self, feature, step: float | None = None, **kwargs):
        """Get feature distribution."""
        if isinstance(feature, dict):
            return {k: self._get_distribution(k, step, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self._get_distribution(k, step) for k in feature]

        return self._get_distribution(feature, step, **kwargs)

    def _get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]:
        calc = self._gen_feature_calculator(feature)
        feat = self._calc_feature(calc, **kwargs)
        return feat

    def _get_distribution(
        self, feature: Feature, step: float | None, **kwargs
    ) -> npt.NDArray[np.int32]:
        if callable(calc := getattr(self, f"get_{feature}_distribution", None)):
            return calc  # custom feature distribution

        feat = self._get(feature, **kwargs)
        step = cast(float, feat.max() / 100) if step is None else step
        distribution = to_distribution(feat, step)
        return distribution

    def _gen_feature_calculator(self, feature: Feature) -> Callable[[], npt.NDArray]:
        if callable(calc := getattr(self, f"get_{feature}", None)):
            return calc  # custom features

        components = feature.split("_")
        if (module := getattr(self, f"_{components[0]}_features", None)) and callable(
            calc := getattr(module, f"get_{'_'.join(components[1:])}", None)
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

    # Feature distribution

    def get_radial_distance_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._node_features.get_radial_distance_distribution(**kwargs)

    def get_node_branch_order_distribution(self, **kwargs) -> npt.NDArray[np.int32]:
        return self._node_features.get_branch_order_distribution(**kwargs)

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
