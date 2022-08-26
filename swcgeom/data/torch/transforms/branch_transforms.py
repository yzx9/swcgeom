from typing import Callable, Literal

import torch

from ....core import Branch
from ...transforms import Transform


class BranchResampler(Transform[Branch, Branch]):
    """Resample branch."""

    def __init__(self, num: int) -> None:
        super().__init__()
        self.num = num

    def __call__(self, x: Branch) -> Branch:
        return x.resample("linear", num=self.num)

    def get_name(self) -> str:
        return f"resample{self.num}"


class BranchStandardize(Transform[Branch, Branch]):
    """Standarize branch."""

    def __call__(self, x: Branch) -> Branch:
        return x.standardize()

    def get_name(self) -> str:
        return "standardized"


class BranchToTensor(Transform[Branch, torch.Tensor]):
    """Transform branch to ~torch.Tensor."""

    channels: str
    channel_first: bool

    fn: Callable[[Branch], torch.Tensor]

    def __init__(
        self, channels: Literal["xyz", "xyzr"] = "xyz", channel_first: bool = True
    ):
        """Transform branch to ~torch.Tensor.

        Parameters
        ----------
        channels : str, default `xyz`
            Output channels and order, support `xyz` and `xyzr` now.
        channel_first : bool
            Tensor dimension order. If `True`, the tensor is of shape (C, N), otherwise of
            shape (N, C).
        """

        super().__init__()
        self.channels = channels
        self.channel_first = channel_first

        match self.channels:
            case "xyz":
                self.fn = lambda x: torch.from_numpy(x.xyz()).float()
            case "xyzr":
                self.fn = lambda x: torch.from_numpy(x.xyzr()).float()
            case _:
                raise ValueError("invalid channel.")

    def __call__(self, x: Branch) -> torch.Tensor:
        output = self.fn(x)
        return output.T if self.channel_first else output

    def get_name(self) -> str:
        return "tensor"
