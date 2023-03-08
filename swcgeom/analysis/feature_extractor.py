"""Easy way to compute and visualize common features for feature.

Notes
-----
For development, see method `Features.get_evaluator` to confirm the
naming specification.
"""

from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import Population, Tree
from ..utils import get_fig_ax, padding1d
from .branch_features import BranchFeatures
from .node_features import NodeFeatures
from .path_features import PathFeatures
from .sholl import Sholl

__all__ = ["Feature", "extract_feature"]

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

Bins = int | npt.ArrayLike | str
Range = Optional[Tuple[float, float]]
HistAndBinEdges = Tuple[npt.NDArray, npt.NDArray]
FeatureWithKwargs = Feature | Tuple[Feature, Dict[str, Any]]


class Features:
    """Tree features"""

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

    def get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        feat, kwargs = _get_feature_and_kwargs(feature)
        evaluator = self.get_evaluator(feat)
        feat = evaluator(**kwargs)
        return feat

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

    def get_sholl(self, **kwargs) -> npt.NDArray[np.float32]:
        return Sholl(self.tree, **kwargs).get().astype(np.float32)


class FeatureExtractor:
    # fmt:off
    @overload
    def get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]: ...
    @overload
    def get(self, feature: List[FeatureWithKwargs]) -> List[npt.NDArray[np.float32]]: ...
    @overload
    def get(self, feature: Dict[Feature, Dict[str, Any]]) -> Dict[str, npt.NDArray[np.float32]]: ...
    # fmt:on
    def get(self, feature, **kwargs):
        """Get feature.

        Notes
        -----
        Shape of returned array is not uniform, `TreeFeatureExtractor`
        returns array of shape (L, ), `PopulationFeatureExtracor`
        returns array of shape (N, L).
        """
        if isinstance(feature, dict):
            return {k: self._get(k, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self._get(k) for k in feature]

        return self._get(feature, **kwargs)

    def plot(self, feature: FeatureWithKwargs, **kwargs) -> Axes:  # TODO: sholl
        vals = self._get(feature)
        return self._plot(vals, **kwargs)

    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    def _plot(self, vals: npt.NDArray[np.float32], **kwargs) -> Axes:
        raise NotImplementedError()


class TreeFeatureExtractor(FeatureExtractor):
    """Extract feature from tree."""

    features: Features

    def __init__(self, tree: Tree) -> None:
        super().__init__()
        self.features = Features(tree)

    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        feat, kwargs = _get_feature_and_kwargs(feature, **kwargs)
        return self.features.get(feat, **kwargs)

    def _plot(self, vals: npt.NDArray[np.float32], **kwargs) -> Axes:
        return sns.histplot(x=vals, **kwargs)


class PopulationFeatureExtractor(FeatureExtractor):
    """Extract feature from population."""

    population: Population

    @cached_property
    def _trees(self) -> List[Features]:
        return [Features(t) for t in self.population]

    def __init__(self, population: Population) -> None:
        super().__init__()
        self.population = population

    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        vals = [f.get(feature, **kwargs) for f in self._trees]
        len_max = max(len(v) for v in vals)
        v = np.stack([padding1d(len_max, v, dtype=np.float32) for v in vals])
        return v

    def _plot(self, vals: npt.NDArray[np.float32], **kwargs) -> Axes:
        vals = vals.flatten()
        return sns.histplot(x=vals, **kwargs)


def extract_feature(obj: Tree | Population) -> FeatureExtractor:
    if isinstance(obj, Tree):
        return TreeFeatureExtractor(obj)

    if isinstance(obj, Population):
        return PopulationFeatureExtractor(obj)

    raise TypeError("Invalid argument type.")


def _get_feature_and_kwargs(
    feature: FeatureWithKwargs, **kwargs
) -> Tuple[Feature, Dict[str, Any]]:
    if isinstance(feature, tuple):
        return feature[0], {**feature[1], **kwargs}

    return feature, kwargs
