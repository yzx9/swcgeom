"""Image stack folder."""

import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Literal, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .io import read_imgs

__all__ = [
    "ImageStackFolder",
    "LabeledImageStackFolder",
    "PathImageStackFolder",
]

T = TypeVar("T")


class ImageStackFolderBase(ABC):
    """Image stack folder base."""

    files: List[str]

    def __init__(self, files: Iterable[str]) -> None:
        super().__init__()
        self.files = list(files)

    @abstractmethod
    def __getitem__(self, key: str, /) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.files)

    def _get(self, fname: str) -> npt.NDArray[np.float32]:
        imgs = self.read_imgs(fname)
        # TODO: support transforms
        return imgs

    @staticmethod
    def read_imgs(fname: str) -> npt.NDArray[np.float32]:
        imgs = read_imgs(fname).get_full()
        imgs = np.moveaxis(imgs, -1, 0)  # (X, Y, Z, C) -> (C, X, Y, Z)
        return imgs

    @staticmethod
    def scan(root: str, *, pattern: Optional[str] = None) -> List[str]:
        is_valid = re.compile(pattern).match if pattern is not None else truthly

        fs = []
        for d, _, files in os.walk(root):
            fs.extend(os.path.join(d, f) for f in files if is_valid(f))

        return fs


class ImageStackFolder(ImageStackFolderBase):
    """Image stack folder."""

    def __getitem__(self, idx: int, /) -> npt.NDArray[np.float32]:
        return self._get(self.files[idx])

    @classmethod
    def from_dir(cls, root: str, *, pattern: Optional[str] = None) -> Self:
        return cls(cls.scan(root, pattern=pattern))


class LabeledImageStackFolder(ImageStackFolderBase):
    """Image stack folder with label."""

    labels: List[int]

    def __init__(self, files: Iterable[str], labels: Iterable[int]):
        super().__init__(files)
        self.labels = list(labels)

    def __getitem__(self, idx: int) -> Tuple[npt.NDArray[np.float32], int]:
        return self.read_imgs(self.files[idx]), self.labels[idx]

    @classmethod
    def from_dir(
        cls,
        root: str,
        label: int | Callable[[str], int],
        *,
        pattern: Optional[str] = None,
    ) -> Self:
        files = cls.scan(root, pattern=pattern)
        if callable(label):
            labels = [label(f) for f in files]
        elif isinstance(label, int):
            labels = [label for _ in files]
        else:
            raise ValueError("")
        return cls(files, labels)


class PathImageStackFolder(ImageStackFolder):
    """Image stack folder with relpath."""

    root: str

    def __getitem__(self, idx: int) -> Tuple[npt.NDArray[np.float32], str]:
        relpath = os.path.relpath(self.files[idx], self.root)
        return self.read_imgs(self.files[idx]), relpath

    @classmethod
    def from_dir(cls, root: str, *, pattern: Optional[str] = None) -> Self:
        return cls(cls.scan(root, pattern=pattern))


def truthly(*args, **kwargs) -> Literal[True]:  # pylint: disable=unused-argument
    return True
