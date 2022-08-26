from ...core.branch import Branch
from .transform import Transform


class BranchResampler(Transform[Branch, Branch]):
    def __init__(self, num: int) -> None:
        super().__init__()
        self.num = num

    def __call__(self, x: Branch) -> Branch:
        return x.resample("linear", num=self.num)

    def get_name(self) -> str:
        return f"resample{self.num}"


class BranchStandardize(Transform[Branch, Branch]):
    def __call__(self, x: Branch) -> Branch:
        return x.standardize()

    def get_name(self) -> str:
        return "standardized"
