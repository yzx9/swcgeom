"""Branch Dataset."""

import os
import warnings
from typing import Generic, Iterable, TypeVar, cast

import torch
import torch.utils.data

from ...core import Branch
from ...transforms import Identity, Transform
from ...utils import numpy_err
from .tree_folder_dataset import TreeFolderDataset

__all__ = ["BranchDataset"]

T = TypeVar("T")
identity = Identity[Branch]()


class BranchDataset(torch.utils.data.Dataset, Generic[T]):
    """An easy way to get branches."""

    swc_dir: str
    save: str | None
    transform: Transform[Branch, T]

    branches: list[T]

    def __init__(
        self,
        swc_dir: str,
        save: str | bool = True,
        transform: Transform[Branch, T] = identity,
    ) -> None:
        """Create branch dataset.

        Parameters
        ----------
        swc_dir : str
            Path of SWC file directory.
        save : Union[str, bool], default `True`
            Save branch data to file if not False. If `True`, automatically
            generate file name.
        transfroms : Transfroms[Branch, T], optional
            Branch transformations.

        See Also
        --------
        ~swcgeom.transforms : module
            Preset transform set.
        """

        self.swc_dir = swc_dir
        self.transform = transform

        if isinstance(save, str):
            self.save = save
        elif save:
            self.save = os.path.join(swc_dir, self.get_filename())
        else:
            self.save = None

        if self.save and os.path.exists(self.save):
            self.branches = torch.load(self.save)
            return

        self.branches = self.get_branches()
        if self.save:
            torch.save(self.branches, self.save)

    def __getitem__(self, idx: int) -> T:
        """Get branch."""
        return self.branches[idx]

    def __len__(self) -> int:
        """Get length of branches."""
        return len(self.branches)

    def get_filename(self) -> str:
        """Get filename."""
        trans_name = f"-{self.transform}" if self.transform is not identity else ""
        return f"BranchDataset{trans_name}.pt"

    def get_branches(self) -> list[T]:
        """Get all branches."""
        branches = list[T]()

        with numpy_err(all="raise"):
            for tree in TreeFolderDataset(self.swc_dir):
                try:
                    brs = tree.get_branches()
                    brs = [self.transform(br) for br in brs] if self.transform else brs
                    branches.extend(cast(Iterable[T], brs))
                except Exception as ex:  # pylint: disable=broad-except
                    warnings.warn(
                        f"BranchDataset: skip swc '{tree.source}', got warning from numpy: {ex}"
                    )

        return branches
