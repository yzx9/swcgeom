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
