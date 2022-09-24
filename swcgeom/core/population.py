"""Neuron population is a set of tree."""

import os
from typing import List, cast, overload

from .tree import Tree

__all__ = ["Population"]


class Population:
    """Neuron population."""

    swc_dir: str = ""
    swcs: List[str]
    trees: List[Tree | None]

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

    def __repr__(self) -> str:
        return f"Neuron population in '{self.swc_dir}'"

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
                self.trees[i] = Tree.from_swc(self.swcs[i])

    @staticmethod
    def from_swc(swc_dir: str) -> "Population":
        swcs = Population.find_swcs(swc_dir)
        population = Population(swcs)
        population.swc_dir = swc_dir
        return population

    @staticmethod
    def find_swcs(swc_dir: str, suffix: str = ".swc") -> list[str]:
        """Find all swc files."""
        swcs = list[str]()
        for root, _, files in os.walk(swc_dir):
            files = [f for f in files if os.path.splitext(f)[-1] == suffix]
            swcs.extend([os.path.join(root, f) for f in files])

        return swcs
