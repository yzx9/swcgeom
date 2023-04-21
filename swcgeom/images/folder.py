"""Image stack folder."""

import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, List, Literal, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ..transforms import Identity, Transform
from .io import read_imgs

__all__ = [
    "ImageStackFolder",
    "LabeledImageStackFolder",
    "PathImageStackFolder",
]

T = TypeVar("T")


class ImageStackFolderBase(Generic[T], ABC):
    """Image stack folder base."""

    files: List[str]
    transform: Transform[npt.NDArray[np.float32], T]

    def __init__(
        self,
        files: Iterable[str],
        *,
        transform: Optional[Transform[npt.NDArray[np.float32], T]] = None,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.transform = transform or Identity()  # type: ignore

    @abstractmethod
    def __getitem__(self, key: str, /) -> T:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.files)

    def _get(self, fname: str) -> T:
        imgs = self.read_imgs(fname)
        imgs = self.transform(imgs)
        return imgs

    @staticmethod
    def read_imgs(fname: str) -> npt.NDArray[np.float32]:
        return read_imgs(fname).get_full()

    @staticmethod
    def scan(root: str, *, pattern: Optional[str] = None) -> List[str]:
        is_valid = re.compile(pattern).match if pattern is not None else truthly

        fs = []
        for d, _, files in os.walk(root):
            fs.extend(os.path.join(d, f) for f in files if is_valid(f))

        return fs


class ImageStackFolder(Generic[T], ImageStackFolderBase[T]):
    """Image stack folder."""

    def __getitem__(self, idx: int, /) -> T:
        return self._get(self.files[idx])

    @classmethod
    def from_dir(cls, root: str, *, pattern: Optional[str] = None, **kwargs) -> Self:
        """
        Parameters
        ----------
        root : str
        pattern : str, optional
            Filter files by pattern.
        **kwargs
            Pass to `cls.__init__`
        """
        return cls(cls.scan(root, pattern=pattern), **kwargs)


class LabeledImageStackFolder(Generic[T], ImageStackFolderBase[T]):
    """Image stack folder with label."""

    labels: List[int]

    def __init__(self, files: Iterable[str], labels: Iterable[int], **kwargs):
        super().__init__(files, **kwargs)
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
        **kwargs,
    ) -> Self:
        files = cls.scan(root, pattern=pattern)
        if callable(label):
            labels = [label(f) for f in files]
        elif isinstance(label, int):
            labels = [label for _ in files]
        else:
            raise ValueError("invalid label")
        return cls(files, labels, **kwargs)


class PathImageStackFolder(Generic[T], ImageStackFolder[T]):
    """Image stack folder with relpath."""

    root: str

    def __init__(self, files: Iterable[str], *, root: str, **kwargs):
        super().__init__(files, **kwargs)
        self.root = root

    def __getitem__(self, idx: int) -> Tuple[T, str]:
        relpath = os.path.relpath(self.files[idx], self.root)
        return self._get(self.files[idx]), relpath

    @classmethod
    def from_dir(cls, root: str, *, pattern: Optional[str] = None, **kwargs) -> Self:
        """
        Parameters
        ----------
        root : str
        pattern : str, optional
            Filter files by pattern.
        **kwargs
            Pass to `cls.__init__`
        """
        return cls(cls.scan(root, pattern=pattern), root=root, **kwargs)


def truthly(*args, **kwargs) -> Literal[True]:  # pylint: disable=unused-argument
    return True
