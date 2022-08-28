import os
import warnings
from typing import Generic, Iterable, Optional, TypeVar, cast

import numpy as np

import torch
import torch.utils.data

from ...core import Branch
from .transforms import ToBranchTree, Transforms
from .tree_folder_dataset import TreeFolderDataset

T = TypeVar("T")


class BranchDataset(torch.utils.data.Dataset, Generic[T]):
    """An easy way to get branches."""

    swc_dir: str
    save: str | None
    transforms: Transforms[Branch, T] | None

    branches: list[T]

    def __init__(
        self,
        swc_dir: str,
        save: str | bool = True,
        transforms: Optional[Transforms[Branch, T]] = None,
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
        ~swcgeom.data.torch.transforms : module
            Preset transform set.
        """

        self.swc_dir = swc_dir
        self.transforms = transforms

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

    def __getitem__(self, idx: int) -> tuple[T, int]:
        """Get branch data.

        Returns
        -------
        x : T
            Transformed data.
        y : int
            Label of x.
        """
        return self.branches[idx], 0

    def __len__(self) -> int:
        """Get length of branches."""
        return len(self.branches)

    def get_filename(self) -> str:
        """Get filename."""
        names = self.transforms.get_names() if self.transforms else []
        name = "_".join(["branch_dataset", *names])
        return f"{name}.pt"

    def get_branches(self) -> list[T]:
        """Get all branches."""
        transforms = Transforms(ToBranchTree())
        branch_trees = TreeFolderDataset(self.swc_dir, transforms=transforms)
        branches = list[T]()
        old_settings = np.seterr(all="raise")
        for x, y in branch_trees:
            try:
                brs = x.get_branches()
                brs = self.transforms.apply_batch(brs) if self.transforms else brs
                branches.extend(cast(Iterable[T], brs))
            except Exception as ex:
                warnings.warn(
                    f"BranchDataset: skip swc '{x}', got warning from numpy: {ex}"
                )

        np.seterr(**old_settings)
        return branches
