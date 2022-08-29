from ..core import Branch
from .base import Transform


class BranchResampler(Transform[Branch, Branch]):
    """Resample branch."""

    def __init__(self, num: int) -> None:
        super().__init__(f"resample{self.num}")
        self.num = num

    def apply(self, x: Branch) -> Branch:
        return x.resample("linear", num=self.num)


class BranchStandardize(Transform[Branch, Branch]):
    """Standarize branch."""

    def __init__(self) -> None:
        super().__init__("standardized")

    def apply(self, x: Branch) -> Branch:
        return x.standardize()
