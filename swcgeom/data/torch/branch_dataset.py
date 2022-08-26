import os
import warnings
from typing import Optional

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ... import Branch
from ..transforms import Transforms
from .branch_tree_folder_dataset import BranchTreeFolderDataset


class BranchDataset(Dataset):
    """An easy way to get branches."""

    swc_dir: str
    save: str | None
    transforms: Transforms[Branch, Branch] | None

    branches: list[Tensor]  # list of shape (N, 3)

    def __init__(
        self,
        swc_dir: str,
        save: str | bool = True,
        transforms: Optional[Transforms[Branch, Branch]] = None,
    ) -> None:
        """Create branch dataset.

        Parameters
        ----------
        swc_dir : str
            Path of SWC file directory.
        save : Union[str, bool], default `True`
            Save branch data to file if not False. If `True`, automatically
            generate file name.
        standardize : bool, default `True`
            See also ~neuron.Branch.standardize.
        resample : Optional[int], optional
            Resampling branch to N points if not `None`.
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

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Get branch.

        Returns
        -------
        x : ~torch.Tensor
            Tensor of shape (3, branch_length).
        y : int
            Label of x.
        """
        return self.branches[idx], 0

    def __len__(self) -> int:
        """Get length of branches."""
        return len(self.branches)

    def get_filename(self) -> str:
        names = self.transforms.get_names() if self.transforms else []
        name = "_".join(["branch_dataset", *names])
        return f"{name}.pt"

    def get_branches(self) -> list[Tensor]:
        """Get all branches."""
        branch_trees = BranchTreeFolderDataset(self.swc_dir)
        branches = list[Branch]()
        old_settings = np.seterr(all="raise")
        for x, y in branch_trees:
            try:
                brs = x.get_branches()
                brs = self.transforms.apply_batch(brs) if self.transforms else brs
                branches.extend(brs)
            except Exception as ex:
                warnings.warn(
                    f"BranchDataset: skip swc '{x}', got warning from numpy: {ex}"
                )

        np.seterr(**old_settings)
        return [torch.from_numpy(br.xyz().T).float() for br in branches]
