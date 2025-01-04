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


"""Transformation in path."""

from swcgeom.core import Path, Tree, redirect_tree
from swcgeom.transforms.base import Transform

__all__ = ["PathToTree", "PathReverser"]


class PathToTree(Transform[Path, Tree]):
    """Transform path to tree."""

    def __call__(self, x: Path) -> Tree:
        t = Tree(
            x.number_of_nodes(),
            type=x.type(),
            id=x.id(),
            x=x.x(),
            y=x.y(),
            z=x.z(),
            r=x.r(),
            pid=x.pid(),
            source=x.source,
            comments=x.comments.copy(),
            names=x.names,
        )
        return t


class PathReverser(Transform[Path, Path]):
    r"""Reverse path.

    ```text
    a -> b -> ... -> y -> z
    // to
    a <- b <- ... <- y <- z
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.to_tree = PathToTree()

    def __call__(self, x: Path) -> Path:
        x[0].type, x[-1].type = x[-1].type, x[0].type
        t = self.to_tree(x)
        t = redirect_tree(t, x[-1].id)
        p = t.get_paths()[0]
        return p
