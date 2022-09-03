"""Transformations in branch requires pytorch."""

from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import torch

from ...core import Branch
from ...transforms import Transform

__all__ = ["BranchToTensor"]


class BranchToTensor(Transform[Branch, torch.Tensor]):
    """Transform branch to ~torch.Tensor."""

    channels: str
    channel_first: bool

    get_channels: Callable[[Branch], npt.NDArray[np.float32]]

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
                self.get_channels = lambda x: x.xyz()
            case "xyzr":
                self.get_channels = lambda x: x.xyzr()
            case _:
                raise ValueError("unsupported channels.")

    def __repr__(self) -> str:
        return f"BranchToTensor-{self.channels}{'-ChannelFirst' if self.channel_first else ''}"

    def __call__(self, x: Branch) -> torch.Tensor:
        channels = self.get_channels(x)  # (N, C)
        tensor = torch.from_numpy(channels).float()
        return tensor.T if self.channel_first else tensor
