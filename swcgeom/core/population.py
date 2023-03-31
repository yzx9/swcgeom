"""Neuron population is a set of tree."""

import os
import warnings
from functools import reduce
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, cast, overload

import numpy as np
from typing_extensions import Self

from .swc import eswc_cols
from .tree import Tree

__all__ = ["Population", "Populations"]


class Population:
    """Neuron population."""

    root: str
    swcs: List[str]
    trees: List[Tree | None]
    kwargs: Dict[str, Any]

    def __init__(
        self, swcs: List[str], lazy_loading: bool = True, root: str = "", **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.swcs = swcs
        self.trees = [None for _ in swcs]
        self.kwargs = kwargs
        if not lazy_loading:
            self.load(slice(len(swcs)))

    # fmt:off
    @overload
    def __getitem__(self, key: slice) -> List[Tree]: ...
    @overload
    def __getitem__(self, key: int) -> Tree: ...
    # fmt:on
    def __getitem__(self, key):
        self.load(key)
        if isinstance(key, slice):
            return [cast(Tree, self.trees[i]) for i in range(*key.indices(len(self)))]

        if isinstance(key, (int, np.integer)):
            return cast(Tree, self.trees[key])

        raise TypeError("Invalid argument type.")

    def __len__(self) -> int:
        return len(self.swcs)

    def __iter__(self) -> Iterator[Tree]:
        return (self[i] for i in range(self.__len__()))

    def __repr__(self) -> str:
        return f"Neuron population in '{self.root}'"

    # fmt:off
    @overload
    def load(self, key: slice) -> None: ...
    @overload
    def load(self, key: int) -> None: ...
    # fmt:on
    def load(self, key):
        if isinstance(key, slice):
            idx = range(*key.indices(len(self)))
        elif isinstance(key, (int, np.integer)):
            idx = [key]
        else:
            raise TypeError("Invalid argument type.")

        for i in idx:
            if self.trees[i] is None:
                self.trees[i] = Tree.from_swc(self.swcs[i], **self.kwargs)

    @classmethod
    def from_swc(cls, root: str, ext: str = ".swc", **kwargs) -> Self:
        swcs = cls.find_swcs(root, ext)
        if len(swcs) == 0:
            warnings.warn(f"no trees in population from '{root}'")
        return Population(swcs, root=root, **kwargs)

    @classmethod
    def from_eswc(
        cls,
        root: str,
        ext: str = ".eswc",
        extra_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> Self:
        extra_cols = extra_cols or []
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
        self, populations: List[Population], labels: Optional[List[str]] = None
    ) -> None:
        self.len = min(len(p) for p in populations)
        self.populations = populations

        labels = labels or ["" for i in populations]
        assert len(labels) == len(
            populations
        ), f"got {len(populations)} populations, but {len(labels)} labels"
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
        swcs = list(chain.from_iterable(p.swcs for p in self.populations))
        return Population(swcs)

    @classmethod
    def from_swc(  # pylint: disable=too-many-arguments
        cls,
        roots: List[str],
        ext: str = ".swc",
        intersect: bool = True,
        check_same: bool = True,
        labels: Optional[List[str]] = None,
        **kwargs,
    ) -> Self:
        """Get population from dirs.

        Parameters
        ----------
        roots : list of str
        intersect : bool, default `True`
            Take the intersection of these populations.
        check_same : bool, default `True`
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
            assert reduce(lambda a, b: a == b, fs), "not the same among populations"

        populations = [
            Population([os.path.join(d, p) for p in fs[i]], root=d, **kwargs)
            for i, d in enumerate(roots)
        ]
        return cls(populations, labels=labels)

    @classmethod
    def from_eswc(
        cls, roots: List[str], extra_cols: Optional[List[str]] = None, **kwargs
    ) -> Self:
        extra_cols = extra_cols or []
        extra_cols.extend(k for k, t in eswc_cols)
        return cls.from_swc(roots, extra_cols=extra_cols, **kwargs)
