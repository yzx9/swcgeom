"""Tree Folder Dataset."""

from typing import Generic, TypeVar, cast

import torch.utils.data

from ...core import Population, Tree
from ...transforms import Identity, Transform

__all__ = ["TreeFolderDataset"]

T = TypeVar("T")
identity = Identity[Tree]()


class TreeFolderDataset(torch.utils.data.Dataset, Generic[T]):
    """Tree folder in swc format."""

    population: Population
    transform: Transform[Tree, T]

    def __init__(
        self,
        swc_dir: str,
        transform: Transform[Tree, T] = identity,
    ) -> None:
        """Create tree dataset.

        Parameters
        ----------
        swc_dir : str
            Path of SWC file directory.
        transfroms : Transfroms[Tree, T], optional
            Branch transformations.

        See Also
        --------
        ~swcgeom.data.torch.transforms : module
            Preset transform set.
        """
        super().__init__()
        self.population = Population.from_swc(swc_dir)
        self.transform = transform

    def __getitem__(self, idx: int) -> T:
        """Get a tree."""
        tree = self.population[idx]
        x = self.transform(tree) if self.transform is not identity else tree
        return cast(T, x)

    def __len__(self) -> int:
        """Get length of set of trees."""
        return len(self.population)
