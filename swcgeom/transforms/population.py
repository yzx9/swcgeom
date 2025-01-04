# Copyright 2022-2025 Zexin Yuan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Transformation in population."""

from swcgeom.core import Population, Tree
from swcgeom.transforms.base import Transform

__all__ = ["PopulationTransform"]


class PopulationTransform(Transform[Population, Population]):
    """Apply transformation for each tree in population."""

    def __init__(self, transform: Transform[Tree, Tree]):
        super().__init__()
        self.transform = transform

    def __call__(self, population: Population) -> Population:
        trees: list[Tree] = []
        for t in population:
            new_t = self.transform(t)
            if new_t.source == "":
                new_t.source = t.source
            trees.append(new_t)

        return Population(trees, root=population.root)

    def extra_repr(self) -> str:
        return f"transform={self.transform}"
