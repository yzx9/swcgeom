from ..core import Branch
from .base import Transform


class BranchResampler(Transform[Branch, Branch]):
    r"""Resample branch."""

    def __init__(self, num: int) -> None:
        super().__init__()
        self.num = num

    def __call__(self, x: Branch) -> Branch:
        return x.resample("linear", num=self.num)

    def __repr__(self) -> str:
        return f"BranchResampler{self.num}"


class BranchStandardizer(Transform[Branch, Branch]):
    r"""Standarize branch."""

    def __call__(self, x: Branch) -> Branch:
        return x.standardize()


class BranchStandardize(BranchStandardizer):
    r"""Standarize branch.

    .. deprecated:: 0.1.8
       `BranchTreeFolderDataset` will be removed in v0.2.0, because
       this is a typo, use `BranchStandardizer` instead.
    """
    pass
