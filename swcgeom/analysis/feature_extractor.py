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

from ..core import Population, Tree
from ..utils import to_distribution
from .branch_features import BranchFeatures
from .node_features import NodeFeatures
from .path_features import PathFeatures
from .sholl import Sholl

__all__ = ["Feature", "FeatureExtractor", "PopulationFeatureExtractor"]

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


class _Features:
    tree: Tree

    # Modules
    # fmt:off
    @cached_property
    def _node_features(self) -> NodeFeatures: return NodeFeatures(self.tree)
    @cached_property
    def _branch_features(self) -> BranchFeatures: return BranchFeatures(self.tree)
    @cached_property
    def _path_features(self) -> PathFeatures: return PathFeatures(self.tree)
    # fmt:on

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]:
        evaluator = self.get_evaluator(feature)
        feat = evaluator(**kwargs).astype(np.float32)
        return feat

    def get_distribution(
        self, feature: Feature, step: float | None, **kwargs
    ) -> npt.NDArray[np.int32]:
        if callable(calc := getattr(self, f"get_{feature}_distribution", None)):
            return calc  # custom feature distribution

        feat = self.get(feature, **kwargs)
        step = cast(float, feat.max() / 100) if step is None else step
        distribution = to_distribution(feat, step)
        return distribution

    def get_evaluator(self, feature: Feature) -> Callable[[], npt.NDArray]:
        if callable(calc := getattr(self, f"get_{feature}", None)):
            return calc  # custom features

        components = feature.split("_")
        if (module := getattr(self, f"_{components[0]}_features", None)) and callable(
            calc := getattr(module, f"get_{'_'.join(components[1:])}", None)
        ):
            return calc

        raise ValueError(f"Invalid feature: {feature}")

    # Features

    def get_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return np.array([self.tree.length(**kwargs)], dtype=np.float32)

    def get_sholl(self, **kwargs) -> npt.NDArray[np.int32]:
        return Sholl(self.tree, **kwargs).get_count()


class FeatureExtractor:
    """Extract feature from tree."""

    features: _Features

    def __init__(self, tree: Tree) -> None:
        self.features = _Features(tree)

    # fmt:off
    @overload
    def get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]: ...
    @overload
    def get(self, feature: List[Feature], **kwargs) -> List[npt.NDArray[np.float32]]: ...
    @overload
    def get(self, feature: Dict[Feature, Dict[str, Any]], **kwargs) -> Dict[str, npt.NDArray[np.float32]]: ...
    # fmt:on
    def get(self, feature, **kwargs):
        """Get feature."""
        if isinstance(feature, dict):
            return {k: self.features.get(k, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self.features.get(k) for k in feature]

        return self.features.get(feature, **kwargs)

    # fmt:off
    @overload
    def get_distribution(self, feature: Feature, step: float = ..., **kwargs) -> npt.NDArray[np.int32]: ...
    @overload
    def get_distribution(self, feature: List[Feature], step: float = ..., **kwargs) -> List[npt.NDArray[np.int32]]: ...
    @overload
    def get_distribution(self, feature: Dict[Feature, Dict[str, Any]], step: float = ..., **kwargs) -> Dict[str, npt.NDArray[np.int32]]: ...
    # fmt:on
    def get_distribution(self, feature, step: float | None = None, **kwargs):
        """Get feature distribution."""
        if isinstance(feature, dict):
            return {
                k: self.features.get_distribution(k, step, **v)
                for k, v in feature.items()
            }

        if isinstance(feature, list):
            return [self.features.get_distribution(k, step) for k in feature]

        return self.features.get_distribution(feature, step, **kwargs)


class PopulationFeatureExtractor:
    """Extract feature from population."""

    population: Population

    @cached_property
    def _trees(self) -> List[_Features]:
        return [_Features(tree) for tree in self.population]

    def __init__(self, population: Population) -> None:
        self.population = population

    # fmt:off
    @overload
    def get(self, feature: Feature, **kwargs) -> List[npt.NDArray[np.float32]]: ...
    @overload
    def get(self, feature: Dict[Feature, Dict[str, Any]], **kwargs) -> Dict[str, List[npt.NDArray[np.float32]]]: ...
    # fmt:on
    def get(self, feature, **kwargs):
        """Get feature."""
        if isinstance(feature, dict):
            return {k: self._get(k, **v) for k, v in feature.items()}

        return self._get(feature, **kwargs)

    # fmt:off
    @overload
    def get_distribution(self, feature: Feature, step: float = ..., **kwargs) -> npt.NDArray[np.int32]: ...
    @overload
    def get_distribution(self, feature: List[Feature], step: float = ..., **kwargs) -> List[npt.NDArray[np.int32]]: ...
    @overload
    def get_distribution(self, feature: Dict[Feature, Dict[str, Any]], step: float = ..., **kwargs) -> Dict[str, npt.NDArray[np.int32]]: ...
    # fmt:on
    def get_distribution(self, feature, step: float | None = None, **kwargs):
        """Get feature distribution."""
        if isinstance(feature, dict):
            return {k: self._get_distribution(k, step, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self._get_distribution(k, step) for k in feature]

        return self._get_distribution(feature, step, **kwargs)

    def _get(self, feature: Feature, **kwargs) -> List[npt.NDArray[np.float32]]:
        return [ex.get(feature, **kwargs) for ex in self._trees]

    def _get_distribution(
        self, feature: Feature, step: float | None, **kwargs
    ) -> npt.NDArray[np.int32]:
        feat = np.concatenate(self._get(feature, **kwargs))
        step = cast(float, feat.max() / 100) if step is None else step
        distribution = to_distribution(feat, step)
        return distribution
