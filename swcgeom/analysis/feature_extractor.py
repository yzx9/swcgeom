"""Easy way to compute and visualize common features for feature.

Notes
-----
For development, see method `Features.get_evaluator` to confirm the
naming specification.
"""

from functools import cached_property
from itertools import chain
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes

from ..core import Population, Populations, Tree
from ..utils import padding1d
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
        if isinstance(feature, tuple):
            feat, kwargs = feature[0], {**feature[1], **kwargs}
        else:
            feat, kwargs = feature, kwargs

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

    def plot(
        self, feature: FeatureWithKwargs, title: str | bool = True, **kwargs
    ) -> Axes:  # TODO: sholl
        vals = self._get(feature)
        ax = self._plot(vals, **kwargs)

        if isinstance(title, str):
            ax.set_title(title)
        elif title is True:
            title = feature[0] if isinstance(feature, tuple) else feature
            title = title.replace("_", " ").title()
            ax.set_title(title)

        ax.set_ylabel("Count")
        return ax

    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    def _plot(self, vals: npt.NDArray[np.float32], **kwargs) -> Axes:
        raise NotImplementedError()


class TreeFeatureExtractor(FeatureExtractor):
    """Extract feature from tree."""

    _tree: Tree
    _features: Features

    def __init__(self, tree: Tree) -> None:
        super().__init__()
        self._tree = tree
        self._features = Features(tree)

    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        return self._features.get(feature, **kwargs)

    def _plot(self, vals: npt.NDArray[np.float32], **kwargs) -> Axes:
        return sns.histplot(x=vals, **kwargs)


class PopulationFeatureExtractor(FeatureExtractor):
    """Extract feature from population."""

    _population: Population
    _features: List[Features]

    def __init__(self, population: Population) -> None:
        super().__init__()
        self._population = population
        self._features = [Features(t) for t in self._population]

    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        vals = [f.get(feature, **kwargs) for f in self._features]
        len_max = max(len(v) for v in vals)
        v = np.stack([padding1d(len_max, v, dtype=np.float32) for v in vals])
        return v

    def _plot(
        self, vals: npt.NDArray[np.float32], bins="auto", range=None, **kwargs
    ) -> Axes:
        bin_edges = np.histogram_bin_edges(vals, bins, range)
        hists = [
            np.histogram(v, bins=bin_edges, weights=(v != 0).astype(np.int32))[0]
            for v in vals
        ]
        hist = np.concatenate(hists)
        x = np.tile((bin_edges[:-1] + bin_edges[1:]) / 2, len(self._population))
        return sns.lineplot(x=x, y=hist, **kwargs)


class PopulationsFeatureExtractor(FeatureExtractor):
    """Extract feature from population."""

    _populations: Populations
    _features: List[List[Features]]

    def __init__(self, populations: Populations) -> None:
        super().__init__()
        self._populations = populations
        self._features = [
            [Features(t) for t in p] for p in self._populations.populations
        ]

    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        vals = [[f.get(feature, **kwargs) for f in fs] for fs in self._features]
        len_max1 = max(len(v) for v in vals)
        len_max2 = max(*chain.from_iterable(((len(vv) for vv in v) for v in vals)))
        out = np.zeros((len(vals), len_max1, len_max2), dtype=np.float32)
        for i, v in enumerate(vals):
            for j, vv in enumerate(v):
                out[i, j, : len(vv)] = vv

        return out

    def _plot(
        self, vals: npt.NDArray[np.float32], bins="auto", range=None, **kwargs
    ) -> Axes:
        bin_edges = np.histogram_bin_edges(vals, bins, range)
        hists = [
            [
                np.histogram(t, bins=bin_edges, weights=(t != 0).astype(np.int32))[0]
                for t in p
            ]
            for p in vals
        ]
        hist = np.concatenate(hists).flatten()

        repeats = np.prod(vals.shape[:2]).item()
        x = np.tile((bin_edges[:-1] + bin_edges[1:]) / 2, repeats)

        labels = self._populations.labels
        length = (len(bin_edges) - 1) * vals.shape[1]
        hue = np.concatenate([np.full(length, fill_value=i) for i in labels])

        return sns.lineplot(x=x, y=hist, hue=hue, **kwargs)


def extract_feature(obj: Tree | Population) -> FeatureExtractor:
    if isinstance(obj, Tree):
        return TreeFeatureExtractor(obj)

    if isinstance(obj, Population):
        return PopulationFeatureExtractor(obj)

    if isinstance(obj, Populations):
        return PopulationsFeatureExtractor(obj)

    raise TypeError("Invalid argument type.")
