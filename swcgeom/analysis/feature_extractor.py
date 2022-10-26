"""A seires of common feature.

Notes
-----
For development, see method `_Features.get_evaluator`
to confirm the naming specification.
"""

from functools import cached_property
from typing import Any, Callable, Dict, List, Literal, Tuple, cast, overload

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import Population, Tree
from ..utils import XYPair, get_fig_ax, padding1d, to_distribution
from .branch_features import BranchFeatures
from .node_features import NodeFeatures
from .path_features import PathFeatures
from .sholl import Sholl

__all__ = [
    "Feature",
    "FeatureExtractor",
    "PopulationFeatureExtractor",
    "extract_feature",
]

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

    def get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]:
        evaluator = self.get_evaluator(feature)
        feat = evaluator(**kwargs).astype(np.float32)
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

    def get_distribution(
        self, feature: Feature, step: float | None, **kwargs
    ) -> XYPair:
        if callable(method := getattr(self, f"get_{feature}_distribution", None)):
            if step is not None:
                kwargs.setdefault("step", step)
            return method(**kwargs)  # custom feature distribution

        feat = self.get(feature, **kwargs)
        step = cast(float, feat.max() / 100) if step is None else step
        x, y = to_distribution(feat, step)
        return x, y

    # Features

    def get_length(self, **kwargs) -> npt.NDArray[np.float32]:
        return np.array([self.tree.length(**kwargs)], dtype=np.float32)

    def get_sholl(self, **kwargs) -> npt.NDArray[np.int32]:
        return Sholl(self.tree, **kwargs).get_count()

    def get_sholl_distribution(self, **kwargs) -> XYPair:
        x, y = Sholl(self.tree, **kwargs).get_distribution()
        return x, y.astype(np.float32)


class FeatureExtractor:
    """Extract feature from tree."""

    features: Features

    def __init__(self, tree: Tree) -> None:
        self.features = Features(tree)

    # fmt:off
    @overload
    def get(self, feature: Feature, **kwargs) -> npt.NDArray[np.float32]: ...
    @overload
    def get(self, feature: List[Feature], **kwargs) -> List[npt.NDArray[np.float32]]: ...
    @overload
    def get(self, feature: Dict[Feature, Dict[str, Any]], **kwargs) -> Dict[str, npt.NDArray[np.float32]]: ...
    # fmt:on
    def get(self, feature, **kwargs):
        """Get feature of shape (L,)."""
        if isinstance(feature, dict):
            return {k: self.features.get(k, **v) for k, v in feature.items()}

        if isinstance(feature, list):
            return [self.features.get(k) for k in feature]

        return self.features.get(feature, **kwargs)

    # fmt:off
    @overload
    def get_distribution(self, feature: Feature, step: float = ..., **kwargs) -> XYPair: ...
    @overload
    def get_distribution(self, feature: List[Feature], step: float = ..., **kwargs) -> List[XYPair]: ...
    @overload
    def get_distribution(self, feature: Dict[Feature, Dict[str, Any]], step: float = ..., **kwargs) -> Dict[str, XYPair]: ...
    # fmt:on
    def get_distribution(self, feature, step: float | None = None, **kwargs):
        """Get feature distribution of shape (S,)."""
        if isinstance(feature, dict):
            return {
                k: self.features.get_distribution(k, step, **v)
                for k, v in feature.items()
            }

        if isinstance(feature, list):
            return [self.features.get_distribution(k, step) for k in feature]

        return self.features.get_distribution(feature, step, **kwargs)

    def plot_distribution(
        self,
        feature: Feature,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        # pylint: disable=unpacking-non-sequence
        x, y = self.get_distribution(feature, **kwargs)
        fig, ax = get_fig_ax(fig, ax)
        sns.lineplot(x=x, y=y, ax=ax)
        return fig, ax


class PopulationFeatureExtractor:
    """Extract feature from population."""

    population: Population

    @cached_property
    def _trees(self) -> List[Features]:
        return [Features(tree) for tree in self.population]

    def __init__(self, population: Population) -> None:
        self.population = population

    # fmt:off
    @overload
    def get(self, feature: Feature, **kwargs) -> List[npt.NDArray[np.float32]]: ...
    @overload
    def get(self, feature: Dict[Feature, Dict[str, Any]], **kwargs) -> Dict[str, List[npt.NDArray[np.float32]]]: ...
    # fmt:on
    def get(self, feature, **kwargs):
        """Get feature list of array of shape (N, L_i).

        Which N is the number of tree of population, L is length of
        nodes.
        """
        if isinstance(feature, dict):
            return {k: self._get(k, **v) for k, v in feature.items()}

        return self._get(feature, **kwargs)

    # fmt:off
    @overload
    def get_distribution(self, feature: Feature, step: float = ..., **kwargs) -> XYPair: ...
    @overload
    def get_distribution(self, feature: List[Feature], step: float = ..., **kwargs) -> List[XYPair]: ...
    @overload
    def get_distribution(self, feature: Dict[Feature, Dict[str, Any]], step: float = ..., **kwargs) -> Dict[str, XYPair]: ...
    # fmt:on
    def get_distribution(self, feature, step: float | None = None, **kwargs):
        """Get feature distribution of shape (N, S).

        Which N is the number of tree of population, S is size of
        distrtibution.

        Returns
        -------
        x : npt.NDArray[np.float32]
            Array of shape (S,).
        y : npt.NDArray[np.float32]
            Array of shape (N, S).
        """
        if isinstance(feature, dict):
            return {
                k: self._get_distribution(k, step=step, **v) for k, v in feature.items()
            }

        if isinstance(feature, list):
            return [self._get_distribution(k, step=step) for k in feature]

        return self._get_distribution(feature, step=step, **kwargs)

    def plot_distribution(
        self,
        feature: Feature,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        # pylint: disable=unpacking-non-sequence
        x, y = self.get_distribution(feature, **kwargs)
        x, y = np.tile(x, y.shape[0]), y.flatten()

        fig, ax = get_fig_ax(fig, ax)
        sns.lineplot(x=x, y=y, ax=ax)
        return fig, ax

    def _get(self, feature: Feature, **kwargs) -> List[npt.NDArray[np.float32]]:
        return [ex.get(feature, **kwargs) for ex in self._trees]

    def _get_distribution(self, feature: Feature, **kwargs) -> XYPair:
        assert len(self._trees) != 0

        x, ys = np.array([], dtype=np.float32), list[npt.NDArray[np.float32]]()
        for features in self._trees:
            xx, y = features.get_distribution(feature, **kwargs)
            x = xx if xx.shape[0] > x.shape[0] else x
            ys.append(y)

        max_len_y = max(y.shape[0] for y in ys)
        y = np.stack([padding1d(max_len_y, y, 0) for y in ys])
        return x, y


# fmt: off
@overload
def extract_feature(obj: Tree) -> FeatureExtractor: ...
@overload
def extract_feature(obj: Population) -> PopulationFeatureExtractor: ...
# fmt: on
def extract_feature(obj):
    if isinstance(obj, Tree):
        return FeatureExtractor(obj)

    if isinstance(obj, Population):
        return PopulationFeatureExtractor(obj)

    raise TypeError("Invalid argument type.")
