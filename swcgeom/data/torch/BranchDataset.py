import os
from warnings import warn

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ... import Branch
from .BranchTreeFolderDataset import BranchTreeFolderDataset


class BranchDataset(Dataset):
    """An easy way to get branches."""

    swc_dir: str
    save: str | None
    resample: int | None
    standardize: bool

    branches: list[Tensor]

    def __init__(
        self,
        swc_dir: str,
        save: str | bool = True,
        standardize: bool = True,
        resample: int | None = 20,
    ) -> None:
        """Create branch dataset.

        Parameters
        ==========
        swc_dir : str.
            Path of SWC file directory.
        save : Union[str, bool], default to `True`.
            Save branch data to file if not False. If `True`, automatically
            generate file name.
        standardize : bool, default to `True`.
            See also ~neuron.Branch.standardize.
        resample : Union[int, None], default to `None`.
            Resampling branch to N points if not `None`.
        """
        self.swc_dir = swc_dir
        self.standardize = standardize
        self.resample = resample

        if save != False:
            if isinstance(save, str):
                self.save = save
            else:
                self.save = os.path.join(
                    swc_dir,
                    "branches"
                    + (f"_resample{resample}" if resample is not None else "")
                    + ("_standardized" if standardize else "")
                    + ".pt",
                )

            if os.path.exists(os.path.join(swc_dir, self.save)):
                self.branches = torch.load(self.save)
                return
        else:
            self.save = None

        self.branches = self.get_branches(swc_dir, standardize, resample)
        if self.save is not None:
            torch.save(self.branches, self.save)

    def __len__(self) -> int:
        """Get length of branches."""
        return len(self.branches)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Get branch.

        Returns
        =======
        x : ~torch.Tensor.
            Tensor shape of (3, branch_length).
        y : int.
            Label of x.
        """
        return self.branches[idx], 0

    @classmethod
    def get_branches(
        cls, swc_dir: str, standardize: bool = True, resample: int | None = None
    ) -> list[Tensor]:
        """Get all branches."""
        branchTrees = BranchTreeFolderDataset(swc_dir)
        branches = list[Branch]()
        old_settings = np.seterr(all="raise")
        for x, y in branchTrees:
            try:
                brs = x.get_branches()

                if standardize == True:
                    brs = [br.standardize() for br in brs]

                if resample is not None:
                    brs = [br.resample("linear", num=resample) for br in brs]

                branches.extend(brs)
            except Exception as ex:
                warn(f"BranchDataset: skip swc '{x}', got warning from numpy: {ex}")

        np.seterr(**old_settings)
        return [torch.from_numpy(br.xyz().T).float() for br in branches]
