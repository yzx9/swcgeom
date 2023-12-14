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
from typing import Any, Callable, Dict, List, Literal, Tuple, overload

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes

from swcgeom.analysis.branch_features import BranchFeatures
from swcgeom.analysis.node_features import (
    BifurcationFeatures,
    NodeFeatures,
    TipFeatures,
)
from swcgeom.analysis.path_features import PathFeatures
from swcgeom.analysis.sholl import Sholl
from swcgeom.analysis.volume import get_volume
from swcgeom.core import Population, Populations, Tree
from swcgeom.utils import padding1d

__all__ = ["Feature", "extract_feature"]

Feature = Literal[
    "length",
    "volume",
    "sholl",
    # node
    "node_count",
    "node_radial_distance",
    "node_branch_order",
    # bifurcation nodes
    "bifurcation_count",
    "bifurcation_radial_distance",
    # tip nodes
    "tip_count",
    "tip_radial_distance",
    # branch
    "branch_length",
    "branch_tortuosity",
    # path
    "path_length",
    "path_tortuosity",
]

NDArrayf32 = npt.NDArray[np.float32]
FeatAndKwargs = Feature | Tuple[Feature, Dict[str, Any]]

Feature1D = set(["length", "volume", "node_count", "bifurcation_count", "tip_count"])


class Features:
    """Tree features"""

    tree: Tree

    # fmt:off
    # Modules
    @cached_property
    def node_features(self) -> NodeFeatures: return NodeFeatures(self.tree)
    @cached_property
    def bifurcation_features(self) -> BifurcationFeatures: return BifurcationFeatures(self.node_features)
    @cached_property
    def tip_features(self) -> TipFeatures: return TipFeatures(self.node_features)
    @cached_property
    def branch_features(self) -> BranchFeatures: return BranchFeatures(self.tree)
    @cached_property
    def path_features(self) -> PathFeatures: return PathFeatures(self.tree)

    # Caches
    @cached_property
    def sholl(self) -> Sholl: return Sholl(self.tree)
    # fmt:on

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get(self, feature: FeatAndKwargs, **kwargs) -> NDArrayf32:
        feat, kwargs = _get_feat_and_kwargs(feature, **kwargs)
        evaluator = self.get_evaluator(feat)
        return evaluator(**kwargs)

    def get_evaluator(self, feature: Feature) -> Callable[[], npt.NDArray]:
        if callable(calc := getattr(self, f"get_{feature}", None)):
            return calc  # custom features

        components = feature.split("_")
        if (module := getattr(self, f"{components[0]}_features", None)) and callable(
            calc := getattr(module, f"get_{'_'.join(components[1:])}", None)
        ):
            return calc

        raise ValueError(f"Invalid feature: {feature}")

    # Custom Features

    def get_length(self, **kwargs) -> NDArrayf32:
        return np.array([self.tree.length(**kwargs)], dtype=np.float32)

    def get_volume(self, **kwargs) -> NDArrayf32:
        return np.array([get_volume(self.tree, **kwargs)], dtype=np.float32)

    def get_sholl(self, **kwargs) -> NDArrayf32:
        return self.sholl.get(**kwargs).astype(np.float32)


class FeatureExtractor(ABC):
    """Extract features from tree."""

    # fmt:off
    @overload
    def get(self, feature: Feature, **kwargs) -> NDArrayf32: ...
    @overload
    def get(self, feature: List[FeatAndKwargs]) -> List[NDArrayf32]: ...
    @overload
    def get(self, feature: Dict[Feature, Dict[str, Any]]) -> Dict[str, NDArrayf32]: ...
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

    def plot(self, feature: FeatAndKwargs, title: str | bool = True, **kwargs) -> Axes:
        """Plot feature with appropriate way.

        Notes
        -----
        The drawing method is different in different classes, different
        in different features, and may different between versions,
        there are NO guarantees.
        """
        feat, feat_kwargs = _get_feat_and_kwargs(feature)
        if callable(custom_plot := getattr(self, f"plot_{feat}", None)):
            ax = custom_plot(feat_kwargs, **kwargs)
        elif feat in Feature1D:
            ax = self._plot_1d(feature, **kwargs)
        else:
            ax = self._plot_histogram(feature, **kwargs)  # default plot

        if isinstance(title, str):
            ax.set_title(title)
        elif title is True:
            ax.set_title(_get_feature_name(feat))

        return ax

    # Custom Plots

    def plot_node_branch_order(self, feature_kwargs: Dict[str, Any], **kwargs) -> Axes:
        vals = self._get("node_branch_order", **feature_kwargs)
        bin_edges = np.arange(int(np.ceil(vals.max() + 1))) + 0.5
        return self._plot_histogram_impl(vals, bin_edges, **kwargs)

    # Implements

    def _get(self, feature: FeatAndKwargs, **kwargs) -> NDArrayf32:
        feat, kwargs = _get_feat_and_kwargs(feature, **kwargs)
        if callable(custom_get := getattr(self, f"get_{feat}", None)):
            return custom_get(**kwargs)

        return self._get_impl(feat, **kwargs)  # default

    def _plot_1d(self, feature: FeatAndKwargs, **kwargs) -> Axes:
        vals = self._get(feature)
        ax = self._plot_1d_impl(vals, **kwargs)
        ax.set_ylabel(_get_feature_name(feature))
        return ax

    def _plot_histogram(
        self,
        feature: FeatAndKwargs,
        bins=20,
        range=None,  # pylint: disable=redefined-builtin
        **kwargs,
    ) -> Axes:
        vals = self._get(feature)
        bin_edges = np.histogram_bin_edges(vals, bins, range)
        return self._plot_histogram_impl(vals, bin_edges, **kwargs)

    @abstractmethod
    def _get_impl(self, feature: Feature, **kwargs) -> NDArrayf32:
        raise NotImplementedError()

    @abstractmethod
    def _plot_1d_impl(self, vals: NDArrayf32, **kwargs) -> Axes:
        raise NotImplementedError()

    @abstractmethod
    def _plot_histogram_impl(
        self, vals: NDArrayf32, bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        raise NotImplementedError()


class TreeFeatureExtractor(FeatureExtractor):
    """Extract feature from tree."""

    _tree: Tree
    _features: Features

    def __init__(self, tree: Tree) -> None:
        super().__init__()
        self._tree = tree
        self._features = Features(tree)

    # Custom Features

    def get_sholl(self, **kwargs) -> NDArrayf32:
        return self._features.sholl.get(**kwargs).astype(np.float32)

    # Custom Plots

    def plot_sholl(
        self,
        feature_kwargs: Dict[str, Any],  # pylint: disable=unused-argument
        **kwargs,
    ) -> Axes:
        _, ax = self._features.sholl.plot(**kwargs)
        return ax

    # Implements

    def _get_impl(self, feature: Feature, **kwargs) -> NDArrayf32:
        return self._features.get(feature, **kwargs)

    def _plot_histogram_impl(
        self, vals: NDArrayf32, bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        hist, _ = np.histogram(vals[vals != 0], bins=bin_edges)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax: Axes = sns.barplot(x=x, y=hist, **kwargs)
        ax.set_ylabel("Count")
        return ax

    def _plot_1d_impl(self, vals: NDArrayf32, **kwargs) -> Axes:
        name = basename(self._tree.source)
        return sns.barplot(x=[name], y=vals.squeeze(), **kwargs)


class PopulationFeatureExtractor(FeatureExtractor):
    """Extract features from population."""

    _population: Population
    _features: List[Features]

    def __init__(self, population: Population) -> None:
        super().__init__()
        self._population = population
        self._features = [Features(t) for t in self._population]

    # Custom Features

    def get_sholl(self, **kwargs) -> NDArrayf32:
        vals, _ = self._get_sholl_impl(**kwargs)
        return vals

    # Custom Plots

    def plot_sholl(self, feature_kwargs: Dict[str, Any], **kwargs) -> Axes:
        vals, rs = self._get_sholl_impl(**feature_kwargs)
        ax = self._lineplot(xs=rs, ys=vals.flatten(), **kwargs)
        ax.set_ylabel("Count of Intersections")
        return ax

    # Implements

    def _get_impl(self, feature: Feature, **kwargs) -> NDArrayf32:
        vals = [f.get(feature, **kwargs) for f in self._features]
        len_max = max(len(v) for v in vals)
        v = np.stack([padding1d(len_max, v, dtype=np.float32) for v in vals])
        return v

    def _get_sholl_impl(
        self, steps: int = 20, **kwargs
    ) -> Tuple[NDArrayf32, NDArrayf32]:
        rmax = max(t.sholl.rmax for t in self._features)
        rs = Sholl.get_rs(rmax=rmax, steps=steps)
        vals = self._get_impl("sholl", steps=rs, **kwargs)
        return vals, rs

    def _plot_histogram_impl(
        self, vals: NDArrayf32, bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        def hist(v):
            return np.histogram(v[v != 0], bins=bin_edges)[0]

        xs = (bin_edges[:-1] + bin_edges[1:]) / 2
        ys = np.stack([hist(v) for v in vals])

        ax: Axes = self._lineplot(xs, ys, **kwargs)
        ax.set_ylabel("Count")
        return ax

    def _plot_1d_impl(self, vals: NDArrayf32, **kwargs) -> Axes:
        x = [basename(t.source) for t in self._population]
        y = vals.flatten()
        ax: Axes = sns.barplot(x=x, y=y, **kwargs)
        ax.axhline(y=y.mean(), ls="--", lw=1)
        ax.set_xticks([])
        return ax

    def _lineplot(self, xs, ys, **kwargs) -> Axes:
        xs = np.tile(xs, len(self._population))
        ys = ys.flatten()
        ax: Axes = sns.lineplot(x=xs, y=ys, **kwargs)
        return ax


class PopulationsFeatureExtractor(FeatureExtractor):
    """Extract feature from population."""

    _populations: Populations
    _features: List[List[Features]]

    def __init__(self, populations: Populations) -> None:
        super().__init__()
        self._populations = populations
        self._features = [[Features(t) for t in p] for p in populations.populations]

    # Custom Features

    def get_sholl(self, **kwargs) -> NDArrayf32:
        vals, _ = self._get_sholl_impl(**kwargs)
        return vals

    # Custom Plots

    def plot_sholl(self, feature_kwargs: Dict[str, Any], **kwargs) -> Axes:
        vals, rs = self._get_sholl_impl(**feature_kwargs)
        ax = self._lineplot(xs=rs, ys=vals, **kwargs)
        ax.set_ylabel("Count of Intersections")
        return ax

    # Implements

    def _get_impl(self, feature: Feature, **kwargs) -> NDArrayf32:
        vals = [[f.get(feature, **kwargs) for f in fs] for fs in self._features]
        len_max1 = max(len(v) for v in vals)
        len_max2 = max(*chain.from_iterable(((len(vv) for vv in v) for v in vals)))
        out = np.zeros((len(vals), len_max1, len_max2), dtype=np.float32)
        for i, v in enumerate(vals):
            for j, vv in enumerate(v):
                out[i, j, : len(vv)] = vv

        return out

    def _get_sholl_impl(
        self, steps: int = 20, **kwargs
    ) -> Tuple[NDArrayf32, NDArrayf32]:
        rmaxs = chain.from_iterable((t.sholl.rmax for t in p) for p in self._features)
        rmax = max(rmaxs)
        rs = Sholl.get_rs(rmax=rmax, steps=steps)
        vals = self._get_impl("sholl", steps=rs, **kwargs)
        return vals, rs

    def _plot_histogram_impl(
        self, vals: NDArrayf32, bin_edges: npt.NDArray, **kwargs
    ) -> Axes:
        def hist(v):
            return np.histogram(v[v != 0], bins=bin_edges)[0]

        xs = (bin_edges[:-1] + bin_edges[1:]) / 2
        ys = np.stack([np.stack([hist(t) for t in p]) for p in vals])

        ax = self._lineplot(xs=xs, ys=ys, **kwargs)
        ax.set_ylabel("Count")
        return ax

    def _plot_1d_impl(self, vals: NDArrayf32, **kwargs) -> Axes:
        labels = self._populations.labels
        xs = np.concatenate([np.full(vals.shape[1], fill_value=i) for i in labels])
        ys = vals.flatten()

        # The numbers of tree in different populations may not be equal
        valid = ys != 0
        xs, ys = xs[valid], ys[valid]

        ax: Axes = sns.boxplot(x=xs, y=ys, **kwargs)
        return ax

    def _lineplot(self, xs, ys, **kwargs) -> Axes:
        p, t, f = ys.shape
        labels = self._populations.labels  # (P,)
        x = np.tile(xs, p * t)  # (F,) -> (P * T * F)
        y = ys.flatten()  # (P, T, F) -> (P * T * F)
        hue = np.concatenate([np.full(t * f, fill_value=i) for i in labels])

        # The numbers of tree in different populations may not be equal
        valid = np.repeat(np.any(ys != 0, axis=2), f)
        x, y, hue = x[valid], y[valid], hue[valid]

        ax: Axes = sns.lineplot(x=x, y=y, hue=hue, **kwargs)
        ax.set_ylabel("Count")
        return ax


def extract_feature(obj: Tree | Population | Populations) -> FeatureExtractor:
    if isinstance(obj, Tree):
        return TreeFeatureExtractor(obj)

    if isinstance(obj, Population):
        return PopulationFeatureExtractor(obj)

    if isinstance(obj, Populations):
        return PopulationsFeatureExtractor(obj)

    raise TypeError("Invalid argument type.")


def _get_feat_and_kwargs(feature: FeatAndKwargs, **kwargs):
    if isinstance(feature, tuple):
        return feature[0], {**feature[1], **kwargs}

    return feature, kwargs


def _get_feature_name(feature: FeatAndKwargs) -> str:
    feat, _ = _get_feat_and_kwargs(feature)
    return feat.replace("_", " ").title()
