"""Neuron population is a set of tree."""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
from tqdm.contrib.concurrent import process_map
from typing_extensions import Self

from swcgeom.core.swc import eswc_cols
from swcgeom.core.tree import Tree

__all__ = ["LazyLoadingTrees", "ChainTrees", "Population", "Populations"]


T = TypeVar("T")


class Trees(Protocol):
    """Trees protocol support index and len."""

    # fmt: off
    def __getitem__(self, key: int, /) -> Tree: ...
    def __len__(self) -> int: ...
    # fmt: on


class LazyLoadingTrees:
    """Lazy loading trees."""

    swcs: List[str]
    trees: List[Tree | None]
    kwargs: Dict[str, Any]

    def __init__(self, swcs: Iterable[str], **kwargs) -> None:
        """
        Paramters
        ---------
        swcs : List of str
        kwargs : Dict[str, Any]
            Forwarding to `Tree.from_swc`
        """

        super().__init__()
        self.swcs = list(swcs)
        self.trees = [None for _ in swcs]
        self.kwargs = kwargs

    def __getitem__(self, key: int, /) -> Tree:
        idx = _get_idx(key, len(self))
        self.load(idx)
        return cast(Tree, self.trees[idx])

    def __len__(self) -> int:
        return len(self.swcs)

    def __iter__(self) -> Iterator[Tree]:
        return (self[i] for i in range(self.__len__()))

    def load(self, key: int) -> None:
        if self.trees[key] is None:
            self.trees[key] = Tree.from_swc(self.swcs[key], **self.kwargs)


class ChainTrees:
    """Chain trees."""

    trees: List[Trees]
    cumsum: npt.NDArray[np.int64]

    def __init__(self, trees: Iterable[Trees]) -> None:
        super().__init__()
        self.trees = list(trees)
        self.cumsum = np.cumsum([0] + [len(ts) for ts in trees])

    def __getitem__(self, key: int, /) -> Tree:
        i, j = 1, len(self.trees)  # cumsum[0] === 0
        idx = _get_idx(key, len(self))
        while i < j:
            mid = (i + j) // 2
            if self.cumsum[mid] <= idx:
                i = mid + 1
            else:
                j = mid

        return self.trees[i - 1][idx - self.cumsum[i - 1]]

    def __len__(self) -> int:
        return self.cumsum[-1].item()

    def __iter__(self) -> Iterator[Tree]:
        return (self[i] for i in range(self.__len__()))


class Population:
    """Neuron population."""

    trees: Trees

    # fmt: off
    @overload
    def __init__(self, swcs: Iterable[str], lazy_loading: bool = ..., root: str = ..., **kwargs) -> None: ...
    @overload
    def __init__(self, trees: Trees, /, *, root: str = "") -> None: ...
    # fmt: on

    def __init__(self, swcs, lazy_loading=True, root="", **kwargs) -> None:
        super().__init__()
        if len(swcs) > 0 and isinstance(swcs[0], str):
            warnings.warn(
                "`Population(swcs)` has been replaced by "
                "`Population(LazyLoadingTrees(swcs))` since v0.8.0 "
                "thus we can create a population from a group of trees, "
                " and this will be removed in next version",
                DeprecationWarning,
            )

            trees = LazyLoadingTrees(swcs, **kwargs)  # type: ignore
            if not lazy_loading:
                for i in range(len(swcs)):
                    trees.load(i)
        else:
            trees = swcs

        self.trees = trees  # type: ignore
        self.root = root

        if len(swcs) == 0:
            warnings.warn(f"no trees in population from '{root}'")

    # fmt:off
    @overload
    def __getitem__(self, key: slice) -> List[Tree]: ...
    @overload
    def __getitem__(self, key: int) -> Tree: ...
    # fmt:on
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [cast(Tree, self.trees[i]) for i in range(*key.indices(len(self)))]

        if isinstance(key, (int, np.integer)):
            return cast(Tree, self.trees[int(key)])

        raise TypeError("Invalid argument type.")

    def __len__(self) -> int:
        return len(self.trees)

    def __iter__(self) -> Iterator[Tree]:
        return (self[i] for i in range(self.__len__()))

    def __repr__(self) -> str:
        return f"Neuron population in '{self.root}'"

    def map(
        self,
        fn: Callable[[Tree], T],
        *,
        max_worker: Optional[int] = None,
        verbose: bool = False,
    ) -> Iterator[T]:
        """Map a function to all trees in the population.

        This is a straightforward interface for parallelizing
        computations. The parameters are intentionally kept simple and
        user-friendly. For more advanced control, consider using
        `concurrent.futures` directly.
        """

        trees = (t for t in self.trees)

        if verbose:
            results = process_map(fn, trees, max_workers=max_worker)
        else:
            with ProcessPoolExecutor(max_worker) as p:
                results = p.map(fn, trees)

        return results

    @classmethod
    def from_swc(cls, root: str, ext: str = ".swc", **kwargs) -> Self:
        if not os.path.exists(root):
            raise FileNotFoundError(
                f"the root does not refers to an existing directory: {root}"
            )

        swcs = cls.find_swcs(root, ext)
        return cls(LazyLoadingTrees(swcs, **kwargs), root=root)

    @classmethod
    def from_eswc(
        cls,
        root: str,
        ext: str = ".eswc",
        extra_cols: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> Self:
        extra_cols = list(extra_cols) if extra_cols is not None else []
        extra_cols.extend(k for k, t in eswc_cols)
        return cls.from_swc(root, ext, extra_cols=extra_cols, **kwargs)

    @staticmethod
    def find_swcs(root: str, ext: str = ".swc", relpath: bool = False) -> List[str]:
        """Find all swc files."""
        swcs: List[str] = []
        for r, _, files in os.walk(root):
            rr = os.path.relpath(r, root) if relpath else r
            fs = filter(lambda f: os.path.splitext(f)[-1] == ext, files)
            swcs.extend(os.path.join(rr, f) for f in fs)

        return swcs


class Populations:
    """A set of population."""

    len: int
    populations: List[Population]
    labels: List[str]

    def __init__(
        self, populations: Iterable[Population], labels: Optional[Iterable[str]] = None
    ) -> None:
        self.len = min(len(p) for p in populations)
        self.populations = list(populations)

        labels = list(labels) if labels is not None else ["" for i in populations]
        assert len(labels) == len(
            self.populations
        ), f"got {len( self.populations)} populations, but has {len(labels)} labels"
        self.labels = labels

    # fmt:off
    @overload
    def __getitem__(self, key: slice) -> List[List[Tree]]: ...
    @overload
    def __getitem__(self, key: int) -> List[Tree]: ...
    # fmt:on
    def __getitem__(self, key):
        return [p[key] for p in self.populations]

    def __len__(self) -> int:
        """Miniumn length of populations."""
        return self.len

    def __iter__(self) -> Iterator[List[Tree]]:
        return (self[i] for i in range(self.len))

    def __repr__(self) -> str:
        return (
            f"A cluster of {self.num_of_populations()} neuron populations, "
            f"each containing at least {self.len} trees"
        )

    def num_of_populations(self) -> int:
        return len(self.populations)

    def to_population(self) -> Population:
        return Population(ChainTrees(p.trees for p in self.populations))

    @classmethod
    def from_swc(
        cls,
        roots: Iterable[str],
        ext: str = ".swc",
        intersect: bool = True,
        check_same: bool = False,
        labels: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> Self:
        """Get population from dirs.

        Parameters
        ----------
        roots : list of str
        intersect : bool, default `True`
            Take the intersection of these populations.
        check_same : bool, default `False`
            Check if the directories contains the same swc.
        labels : List of str, optional
            Label of populations.
        **kwargs : Any
            Forwarding to `Population`.
        """

        fs = [Population.find_swcs(d, ext=ext, relpath=True) for d in roots]
        if intersect:
            inter = list(reduce(lambda a, b: set(a).intersection(set(b)), fs))
            if len(inter) == 0:
                warnings.warn("no intersection among populations")

            fs = [inter for _ in roots]
        elif check_same:
            assert [fs[0] == a for a in fs[1:]], "not the same among populations"

        populations = [
            Population(
                LazyLoadingTrees([os.path.join(d, p) for p in fs[i]], **kwargs), root=d
            )
            for i, d in enumerate(roots)
        ]
        return cls(populations, labels=labels)

    @classmethod
    def from_eswc(
        cls,
        roots: Iterable[str],
        extra_cols: Optional[Iterable[str]] = None,
        *,
        ext: str = ".eswc",
        **kwargs,
    ) -> Self:
        extra_cols = list(extra_cols) if extra_cols is not None else []
        extra_cols.extend(k for k, t in eswc_cols)
        return cls.from_swc(roots, extra_cols=extra_cols, ext=ext, **kwargs)


def _get_idx(key: int, length: int) -> int:
    if key < -length or key >= length:
        raise IndexError(f"The index ({key}) is out of range.")

    if key < 0:  # Handle negative indices
        key += length

    return key
