"""Transformation in branch."""

from ..core import Branch
from .base import Transform

__all__ = ["BranchResampler", "BranchStandardizer"]


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
