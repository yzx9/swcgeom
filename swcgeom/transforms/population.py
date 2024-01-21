"""Transformation in population."""

from typing import List

from swcgeom.core import Population, Tree
from swcgeom.transforms.base import Transform

__all__ = ["PopulationTransform"]


class PopulationTransform(Transform[Population, Population]):
    """Apply transformation for each tree in population."""

    def __init__(self, transform: Transform[Tree, Tree]):
        super().__init__()
        self.transform = transform

    def __call__(self, population: Population) -> Population:
        trees: List[Tree] = []
        for t in population:
            new_t = self.transform(t)
            if new_t.source == "":
                new_t.source = t.source
            trees.append(new_t)

        return Population(trees, root=population.root)

    def extra_repr(self) -> str:
        return f"transform={self.transform}"
