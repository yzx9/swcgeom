
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Neuron population is a set of tree."""

import os
import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from typing import Any, Protocol, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
from tqdm.contrib.concurrent import process_map
from typing import Literal
from typing_extensions import Self

from swcgeom.core.swc import eswc_cols
from swcgeom.core.swc_utils.base import SWCNames
from swcgeom.core.swc_utils.io import read_swc_components
from swcgeom.core.tree import Tree
from swcgeom.utils import PathOrIO

__all__ = ["LazyLoadingTrees", "ChainTrees", "Population", "Populations"]


T = TypeVar("T")


class Trees(Protocol):
    """Trees protocol support index and len."""

    def __getitem__(self, key: int, /) -> Tree: ...
    def __len__(self) -> int: ...


class LazyLoadingTrees:
    """Lazy loading trees."""

    swcs: list[str]
    trees: list[Tree | None]
    kwargs: dict[str, Any]

    def __init__(self, swcs: Iterable[str], **kwargs) -> None:
        """
        Args:
            swcs: List of str
            kwargs: Forwarding to `Tree.from_swc`
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

    trees: list[Trees]
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


class NestTrees:
    def __init__(self, trees: Trees, idx: Iterable[int], /) -> None:
        super().__init__()
        self.trees = trees
        self.idx = list(idx)

    def __getitem__(self, key: int, /) -> Tree:
        return self.trees[self.idx[key]]

    def __len__(self) -> int:
        return len(self.idx)


class Population:
    """Neuron population."""

    trees: Trees

    @overload
    def __init__(
        self, swcs: Iterable[str], lazy_loading: bool = ..., root: str = ..., **kwargs
    ) -> None: ...
    @overload
    def __init__(self, trees: Trees, /, *, root: str = "") -> None: ...
    def __init__(self, swcs, lazy_loading=True, root="", **kwargs) -> None:
        super().__init__()
        if len(swcs) > 0 and isinstance(swcs[0], str):
            warnings.warn(
                "`Population(swcs)` has been replaced by "
                "`Population(LazyLoadingTrees(swcs))` since v0.8.0 thus we can create "
                "a population from a group of trees, and this will be removed in next "
                "version",
                DeprecationWarning,
            )

            trees = LazyLoadingTrees(swcs, **kwargs)
            if not lazy_loading:
                for i in range(len(swcs)):
                    trees.load(i)
        else:
            trees = swcs

        self.trees = trees
        self.root = root

        if len(swcs) == 0:
            warnings.warn(f"no trees in population from '{root}'")

    @overload
    def __getitem__(self, key: slice) -> Trees: ...
    @overload
    def __getitem__(self, key: int) -> Tree: ...
    def __getitem__(self, key: int | slice):
        if isinstance(key, slice):
            trees = NestTrees(self.trees, range(*key.indices(len(self))))
            return cast(Trees, trees)

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
        max_worker: int | None = None,
        verbose: bool = False,
    ) -> Iterator[T]:
        """Map a function to all trees in the population.

        This is a straightforward interface for parallelizing computations. The
        parameters are intentionally kept simple and user-friendly. For more advanced
        control, consider using `concurrent.futures` directly.
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
        extra_cols: Iterable[str] | None = None,
        **kwargs,
    ) -> Self:
        extra_cols = list(extra_cols) if extra_cols is not None else []
        extra_cols.extend(k for k, _ in eswc_cols)
        return cls.from_swc(root, ext, extra_cols=extra_cols, **kwargs)

    @staticmethod
    def find_swcs(root: str, ext: str = ".swc", relpath: bool = False) -> list[str]:
        """Find all swc files."""
        swcs: list[str] = []
        for r, _, files in os.walk(root):
            rr = os.path.relpath(r, root) if relpath else r
            fs = filter(lambda f: os.path.splitext(f)[-1] == ext, files)
            swcs.extend(os.path.join(rr, f) for f in fs)

        return swcs

    @classmethod
    def from_multi_roots_swc(
        cls,
        swc_file: PathOrIO,
        *,
        extra_cols: Iterable[str] | None = None,
        encoding: Literal["detect"] | str = "utf-8",
        names: SWCNames | None = None,
        reset_index_per_subtree: bool = True,
        **kwargs,
    ) -> Self:
        """Create a population from an SWC file containing multiple roots.

        Each root in the SWC file will be treated as the start of a
        separate tree in the resulting population.

        Args:
            swc_file: Path to the SWC file.
            extra_cols: Read more cols in swc file. Passed to
                `read_swc_components`.
            encoding: The name of the encoding used to decode the file.
                Passed to `read_swc_components`.
            names: SWCNames configuration. Passed to `read_swc_components`.
            reset_index_per_subtree: Reset node index for each subtree
                to start with zero. Passed to `read_swc_components`.
            **kwargs: Additional keyword arguments passed to `Tree.from_data_frame`
                for each component tree.

        Returns:
            A Population object where each tree corresponds to a connected
            component from the input SWC file.
        """
        dfs, comments = read_swc_components(
            swc_file,
            extra_cols=extra_cols,
            encoding=encoding,
            names=names,
            reset_index_per_subtree=reset_index_per_subtree,
        )

        trees = [
            Tree.from_data_frame(df, source=f"{swc_file}#component_{i}", comments=comments, **kwargs)
            for i, df in enumerate(dfs)
        ]

        # Use the file path as the 'root' for the population representation
        root_repr = str(swc_file) if not hasattr(swc_file, "name") else swc_file.name
        return cls(trees, root=root_repr)


class Populations:
    """A set of population."""

    len: int
    populations: list[Population]
    labels: list[str]

    def __init__(
        self, populations: Iterable[Population], labels: Iterable[str] | None = None
    ) -> None:
        self.len = min(len(p) for p in populations)
        self.populations = list(populations)

        labels = list(labels) if labels is not None else ["" for i in populations]
        assert len(labels) == len(self.populations), (
            f"got {len(self.populations)} populations, but has {len(labels)} labels"
        )
        self.labels = labels

    @overload
    def __getitem__(self, key: slice) -> list[list[Tree]]: ...
    @overload
    def __getitem__(self, key: int) -> list[Tree]: ...
    def __getitem__(self, key):
        return [p[key] for p in self.populations]

    def __len__(self) -> int:
        """Miniumn length of populations."""
        return self.len

    def __iter__(self) -> Iterator[list[Tree]]:
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
        labels: Iterable[str] | None = None,
        **kwargs,
    ) -> Self:
        """Get population from dirs.

        Args:
            roots: List of str
            intersect: Take the intersection of these populations.
            check_same: Check if the directories contains the same swc.
            labels: Label of populations.
            **kwargs: Forwarding to `Population`.
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
        extra_cols: Iterable[str] | None = None,
        *,
        ext: str = ".eswc",
        **kwargs,
    ) -> Self:
        extra_cols = list(extra_cols) if extra_cols is not None else []
        extra_cols.extend(k for k, _ in eswc_cols)
        return cls.from_swc(roots, extra_cols=extra_cols, ext=ext, **kwargs)


def _get_idx(key: int, length: int) -> int:
    if key < -length or key >= length:
        raise IndexError(f"The index ({key}) is out of range.")

    if key < 0:  # Handle negative indices
        key += length

    return key


# experimental
def filter_population(pop: Population, predicate: Callable[[Tree], bool]) -> Population:
    """Filter trees in the population."""
    # TODO: how to avoid load trees
    idx = [i for i, t in enumerate(pop) if predicate(t)]
    return Population(NestTrees(pop.trees, idx), root=pop.root)
