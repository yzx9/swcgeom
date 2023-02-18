"""Neuron population is a set of tree."""

import os
from itertools import chain
from typing import Any, Dict, Iterator, List, cast, overload

from typing_extensions import Self

from .swc import eswc_cols
from .tree import Tree

__all__ = ["Population", "Populations"]


class Population:
    """Neuron population."""

    root: str = ""
    swcs: List[str]
    trees: List[Tree | None]
    read_kwargs: Dict[str, Any]

    def __init__(self, swcs: List[str], lazy_loading=True) -> None:
        super().__init__()
        self.swcs = swcs
        self.trees = [None for _ in swcs]
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

        if isinstance(key, int):
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
        elif isinstance(key, int):
            idx = [key]
        else:
            raise TypeError("Invalid argument type.")

        for i in idx:
            if self.trees[i] is None:
                self.trees[i] = Tree.from_swc(self.swcs[i], **self.read_kwargs)

    @classmethod
    def from_swc(cls, root: str, suffix: str = ".swc", **kwargs) -> Self:
        swcs = cls.find_swcs(root, suffix)
        population = Population(swcs)
        population.root = root
        population.read_kwargs = kwargs
        return population

    @classmethod
    def from_eswc(
        cls, root: str, suffix: str = ".eswc", extra_cols: List[str] = [], **kwargs
    ) -> Self:
        extra_cols.extend(k for k, t in eswc_cols)
        return cls.from_swc(root, suffix, extra_cols=extra_cols, **kwargs)

    @staticmethod
    def find_swcs(root: str, suffix: str = ".swc") -> List[str]:
        """Find all swc files."""
        swcs: List[str] = []
        for root, _, files in os.walk(root):
            swcs.extend(
                os.path.join(root, f)
                for f in files
                if os.path.splitext(f)[-1] == suffix
            )

        return swcs


class Populations:
    len: int
    populations: List[Population]

    def __init__(self, populations: List[Population]) -> None:
        self.len = min(len(p) for p in self.populations)
        self.populations = populations

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
    def from_swc(cls, *dirs: str, check_same: bool = True, **kwargs) -> Self:
        populations = [Population.from_swc(d, **kwargs) for d in dirs]
        if check_same:
            get_swcs = lambda p: [os.path.relpath(i, p.root) for i in p.swcs]
            t0 = get_swcs(populations[0])
            assert all(
                t0 == get_swcs(p) for p in populations[1:]
            ), "the trees in these population are not the same"

        return cls(populations)

    @classmethod
    def from_eswc(cls, *dirs: str, extra_cols: List[str] = [], **kwargs) -> Self:
        extra_cols.extend(k for k, t in eswc_cols)
        return cls.from_swc(*dirs, extra_cols=extra_cols, **kwargs)
