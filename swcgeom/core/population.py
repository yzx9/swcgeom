"""Neuron population is a set of tree."""

import os
from typing import Any, Dict, Iterator, List, cast, overload

from .swc import eswc_cols
from .tree import Tree

__all__ = ["Population"]


class Population:
    """Neuron population."""

    source: str = ""
    swcs: List[str]
    trees: List[Tree | None]
    read_kwargs: Dict[str, Any]

    def __init__(self, swcs: list[str], lazy_loading=True) -> None:
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
        return f"Neuron population in '{self.source}'"

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

    @staticmethod
    def from_swc(swc_dir: str, suffix: str = ".swc", **kwargs) -> "Population":
        swcs = Population.find_swcs(swc_dir, suffix)
        population = Population(swcs)
        population.source = swc_dir
        population.read_kwargs = kwargs
        return population

    @staticmethod
    def from_eswc(swc_dir: str, suffix: str = ".eswc", **kwargs) -> "Population":
        kwargs.setdefault("extra_cols", [])
        kwargs["extra_cols"].extend(k for k, t in eswc_cols)
        return Population.from_swc(swc_dir, suffix, **kwargs)

    @staticmethod
    def find_swcs(swc_dir: str, suffix: str = ".swc") -> list[str]:
        """Find all swc files."""
        swcs = list[str]()
        for root, _, files in os.walk(swc_dir):
            files = [f for f in files if os.path.splitext(f)[-1] == suffix]
            swcs.extend([os.path.join(root, f) for f in files])

        return swcs
