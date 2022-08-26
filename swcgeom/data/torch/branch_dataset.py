import os
from typing import Optional
from warnings import warn

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ... import Branch
from .branch_tree_folder_dataset import BranchTreeFolderDataset


class BranchDataset(Dataset):
    """An easy way to get branches."""

    swc_dir: str
    save: str | None
    resample: int | None
    standardize: bool

    branches: list[Tensor]  # list of shape (N, 3)

    def __init__(
        self,
        swc_dir: str,
        save: str | bool = True,
        standardize: bool = True,
        resample: Optional[int] = None,
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
        self.standardize = standardize
        self.resample = resample

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
        names = [
            "branches",
            (f"_resample{self.resample}" if self.resample is not None else ""),
            ("_standardized" if self.standardize else ""),
            ".pt",
        ]
        return "".join(names)

    def get_branches(self) -> list[Tensor]:
        """Get all branches."""
        branch_trees = BranchTreeFolderDataset(self.swc_dir)
        branches = list[Branch]()
        old_settings = np.seterr(all="raise")
        for x, y in branch_trees:
            try:
                brs = x.get_branches()

                if self.standardize:
                    brs = map(lambda br: br.standardize(), brs)

                if self.resample:
                    resample = self.resample
                    brs = map(lambda br: br.resample("linear", num=resample), brs)

                branches.extend(brs)
            except Exception as ex:
                warn(f"BranchDataset: skip swc '{x}', got warning from numpy: {ex}")

        np.seterr(**old_settings)
        return [torch.from_numpy(br.xyz().T).float() for br in branches]
