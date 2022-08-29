from typing import Callable, Literal

import torch

from ...core import Branch
from ...transforms import Transform


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

        super().__init__("tensor")
        self.channels = channels
        self.channel_first = channel_first

        match self.channels:
            case "xyz":
                self.fn = lambda x: torch.from_numpy(x.xyz()).float()
            case "xyzr":
                self.fn = lambda x: torch.from_numpy(x.xyzr()).float()
            case _:
                raise ValueError("invalid channel.")

    def apply(self, x: Branch) -> torch.Tensor:
        output = self.fn(x)
        return output.T if self.channel_first else output
