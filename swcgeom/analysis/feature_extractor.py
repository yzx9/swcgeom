"""Easy way to compute and visualize common features for feature.

Notes
-----
For development, see method `Features.get_evaluator` to confirm the
naming specification.
"""

from abc import ABC, abstractmethod
from functools import cached_property
from itertools import chain
from os.path import basename
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes

from ..core import Population, Populations, Tree
from ..utils import padding1d
from .branch_features import BranchFeatures
from .node_features import BifurcationFeatures, NodeFeatures, TipFeatures
from .path_features import PathFeatures
from .sholl import Sholl

__all__ = ["Feature", "extract_feature"]

Feature = Literal[
    "length",
    "sholl",
    # node
    "node_radial_distance",
    "node_branch_order",
    # bifurcation nodes
    "bifurcation_radial_distance",
    "bifurcation_branch_order",
    # tip nodes
    "tip_radial_distance",
    "tip_branch_order",
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
    def _bifurcation_features(self) -> BifurcationFeatures: return BifurcationFeatures(self._node_features)
    @cached_property
    def _tip_features(self) -> TipFeatures: return TipFeatures(self._node_features)
    @cached_property
    def _branch_features(self) -> BranchFeatures: return BranchFeatures(self.tree)
    @cached_property
    def _path_features(self) -> PathFeatures: return PathFeatures(self.tree)
    # fmt:on

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        feat, kwargs = _get_feature_and_kwargs(feature, **kwargs)
        evaluator = self.get_evaluator(feat)
        return evaluator(**kwargs)

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
        return Sholl(self.tree).get(**kwargs).astype(np.float32)


class FeatureExtractor(ABC):
    """Extract features from tree."""

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
            return {k: self._custom_get(k, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self._custom_get(k) for k in feature]

        return self._custom_get(feature, **kwargs)

    def plot(
        self, feature: FeatureWithKwargs, title: str | bool = True, **kwargs
    ) -> Axes:
        """Plot feature with appropriate way.

        Notes
        -----
        The drawing method is different in different classes, different
        in different features, and may different between versions,
        there are NO guarantees.
        """
        feat, _ = _get_feature_and_kwargs(feature)
        if not callable(plot := getattr(self, f"_plot_{feat}", None)):
            plot = self._plot  # default plot

        ax = plot(feature, **kwargs)

        if isinstance(title, str):
            ax.set_title(title)
        elif title is True:
            ax.set_title(feat.replace("_", " ").title())

        return ax

    def _custom_get(
        self, feature: FeatureWithKwargs, **kwargs
    ) -> npt.NDArray[np.float32]:
        feat, _ = _get_feature_and_kwargs(feature)
        if not callable(get := getattr(self, f"_get_{feat}", None)):
            get = self._get  # default

        return get(feature, **kwargs)

    @abstractmethod
    def _get(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    # pylint: disable=redefined-builtin
    def _plot(
        self, feature: FeatureWithKwargs, bins="auto", range=None, **kwargs
    ) -> Axes:
        vals = self._get(feature)
        bin_edges = np.histogram_bin_edges(vals, bins, range)
        return self._plot_histogram(vals, bin_edges, **kwargs)

    @abstractmethod
    def _plot_histogram(
        self, vals: npt.NDArray[np.float32], bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        raise NotImplementedError()

    # custom features

    def _plot_node_branch_order(self, *args, **kwargs) -> Axes:
        return self._impl_plot_node_branch_order(*args, **kwargs)

    def _plot_bifurcation_branch_order(self, *args, **kwargs) -> Axes:
        return self._impl_plot_node_branch_order(*args, **kwargs)

    def _plot_tip_branch_order(self, *args, **kwargs) -> Axes:
        return self._impl_plot_node_branch_order(*args, **kwargs)

    def _plot_sholl(self, feature: FeatureWithKwargs, **kwargs) -> Axes:
        _, feat_kwargs = _get_feature_and_kwargs(feature)
        step = feat_kwargs.get("step", 1)  # TODO: remove hard code
        vals = self._get(feature)
        bin_edges = np.arange(step / 2, step * (vals.shape[-1] + 1), step)
        return self._plot_histogram(vals, bin_edges, **kwargs)

    def _impl_plot_node_branch_order(
        self, feature: FeatureWithKwargs, **kwargs
    ) -> Axes:
        vals = self._get(feature)
        bin_edges = np.arange(int(np.ceil(vals.max() + 1)))
        return self._plot_histogram(vals, bin_edges, **kwargs)


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

    def _plot_histogram(
        self, vals: npt.NDArray[np.float32], bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        weights = (vals != 0).astype(np.int32)
        hist, _ = np.histogram(vals, bins=bin_edges, weights=weights)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax: Axes = sns.barplot(x=x, y=hist, **kwargs)
        ax.set_ylabel("Count")
        return ax

    def _plot_length(self, feature: FeatureWithKwargs, **kwargs) -> Axes:
        name = basename(self._tree.source)
        ax: Axes = sns.barplot(x=[name], y=self._get(feature).squeeze(), **kwargs)
        ax.set_ylabel("Length")
        return ax


class PopulationFeatureExtractor(FeatureExtractor):
    """Extract features from population."""

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

    def _get_sholl(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    def _plot_histogram(
        self, vals: npt.NDArray[np.float32], bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        hists = [
            np.histogram(v, bins=bin_edges, weights=(v != 0).astype(np.int32))[0]
            for v in vals
        ]
        hist = np.concatenate(hists)
        x = np.tile((bin_edges[:-1] + bin_edges[1:]) / 2, len(self._population))

        ax: Axes = sns.lineplot(x=x, y=hist, **kwargs)
        ax.set_ylabel("Count")
        return ax

    def _plot_length(self, feature: FeatureWithKwargs, **kwargs) -> Axes:
        x = [basename(t.source) for t in self._population]
        y = self._get(feature).flatten()
        ax: Axes = sns.barplot(x=x, y=y, **kwargs)
        ax.axhline(y=y.mean(), ls="--", lw=1)
        ax.set_ylabel("Length")
        ax.set_xticks([])
        return ax

    def _plot_sholl(self, feature: FeatureWithKwargs, **kwargs) -> npt.NDArray[np.float32]:
        raise NotImplementedError()


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

    def _plot_histogram(
        self, vals: npt.NDArray[np.float32], bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        def histogram(v):
            return np.histogram(v, bins=bin_edges, weights=(v != 0).astype(np.int32))

        hists = [[histogram(t)[0] for t in p] for p in vals]
        hist = np.concatenate(hists).flatten()

        repeats = np.prod(vals.shape[:2]).item()
        x = np.tile((bin_edges[:-1] + bin_edges[1:]) / 2, repeats)

        labels = self._populations.labels
        length = (len(bin_edges) - 1) * vals.shape[1]
        hue = np.concatenate([np.full(length, fill_value=i) for i in labels])

        ax: Axes = sns.lineplot(x=x, y=hist, hue=hue, **kwargs)
        ax.set_ylabel("Count")
        return ax

    def _plot_length(self, feature: FeatureWithKwargs, **kwargs) -> Axes:
        vals = self._get(feature)
        labels = self._populations.labels
        x = np.concatenate([np.full(vals.shape[1], fill_value=i) for i in labels])
        y = vals.flatten()
        ax: Axes = sns.boxplot(x=x, y=y, **kwargs)
        ax.set_ylabel("Length")
        return ax


def extract_feature(obj: Tree | Population) -> FeatureExtractor:
    if isinstance(obj, Tree):
        return TreeFeatureExtractor(obj)

    if isinstance(obj, Population):
        return PopulationFeatureExtractor(obj)

    if isinstance(obj, Populations):
        return PopulationsFeatureExtractor(obj)

    raise TypeError("Invalid argument type.")


def _get_feature_and_kwargs(feature: FeatureWithKwargs, **kwargs):
    if isinstance(feature, tuple):
        return feature[0], {**feature[1], **kwargs}

    return feature, kwargs
