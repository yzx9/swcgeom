
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Image stack folder."""

import math
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing_extensions import Self, deprecated

from swcgeom.images.io import ScalarType, read_imgs
from swcgeom.transforms import Identity, Transform

__all__ = ["ImageStackFolder", "LabeledImageStackFolder", "PathImageStackFolder"]

T = TypeVar("T")


class ImageStackFolderBase(Generic[ScalarType, T]):
    """Image stack folder base."""

    files: list[str]
    transform: Transform[npt.NDArray[ScalarType], T]

    @overload
    def __init__(
        self,
        files: Iterable[str],
        *,
        transform: Transform[npt.NDArray[np.float32], T] | None = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        files: Iterable[str],
        *,
        dtype: ScalarType,
        transform: Transform[npt.NDArray[ScalarType], T] | None = ...,
    ) -> None: ...
    def __init__(self, files: Iterable[str], *, dtype=None, transform=None) -> None:
        super().__init__()
        self.files = list(files)
        self.dtype = dtype or np.float32
        self.transform = transform or Identity()  # type: ignore

    def __len__(self) -> int:
        return len(self.files)

    def _get(self, fname: str) -> T:
        imgs = self._read(fname)
        imgs = self.transform(imgs)
        return imgs

    def _read(self, fname: str) -> npt.NDArray[ScalarType]:
        return read_imgs(fname, dtype=self.dtype).get_full()  # type: ignore

    @staticmethod
    def scan(root: str, *, pattern: str | None = None) -> list[str]:
        if not os.path.isdir(root):
            raise NotADirectoryError(f"not a directory: {root}")

        is_valid = re.compile(pattern).match if pattern is not None else truthly

        fs = []
        for d, _, files in os.walk(root):
            fs.extend(os.path.join(d, f) for f in files if is_valid(f))

        return fs

    @staticmethod
    @deprecated("Use `~swcgeom.images.io.read_imgs(fname).get_full()` instead")
    def read_imgs(fname: str) -> npt.NDArray[np.float32]:
        """Read images.

        .. deprecated:: 0.16.0
            Use :meth:`~swcgeom.images.io.read_imgs(fname).get_full()` instead.
        """

        return read_imgs(fname).get_full()


@dataclass(frozen=True)
class Statistics:
    count: int = 0
    minimum: float = math.nan
    maximum: float = math.nan
    mean: float = 0
    variance: float = 0


class ImageStackFolder(ImageStackFolderBase[ScalarType, T]):
    """Image stack folder."""

    def __getitem__(self, idx: int, /) -> T:
        return self._get(self.files[idx])

    def stat(self, *, transform: bool = False, verbose: bool = False) -> Statistics:
        """Statistics of folder.

        NOTE: We are asserting that the images are of the same shape.

        Args:
            transform: Apply transform to the images.
                If True, you need to make sure the transformed data is a ndarray.
            verbose: Show progress bar.
        """

        vmin, vmax = math.inf, -math.inf
        n, mean, M2 = 0, None, None

        for idx in tqdm(range(len(self))) if verbose else range(len(self)):
            imgs = self[idx] if transform else self._read(self.files[idx])

            vmin = min(vmin, np.min(imgs))  # type: ignore
            vmax = max(vmax, np.max(imgs))  # type: ignore
            # Welford algorithm to calculate mean and variance
            if mean is None:
                mean = np.zeros_like(imgs)
                M2 = np.zeros_like(imgs)

            n += 1
            delta = imgs - mean
            mean += delta / n
            delta2 = imgs - mean
            M2 += delta * delta2

        if mean is None or M2 is None:  # n = 0
            raise ValueError("empty folder")

        variance = M2 / (n - 1) if n > 1 else np.zeros_like(mean)
        return Statistics(
            count=len(self),
            maximum=vmax,
            minimum=vmin,
            mean=np.mean(mean).item(),
            variance=np.mean(variance).item(),
        )

    @classmethod
    def from_dir(cls, root: str, *, pattern: str | None = None, **kwargs) -> Self:
        """
        Args:
            root: str
            pattern: Filter files by pattern.
            **kwargs: Pass to `cls.__init__`
        """

        return cls(cls.scan(root, pattern=pattern), **kwargs)


class LabeledImageStackFolder(ImageStackFolderBase[ScalarType, T]):
    """Image stack folder with label."""

    labels: list[int]

    def __init__(self, files: Iterable[str], labels: Iterable[int], **kwargs):
        super().__init__(files, **kwargs)
        self.labels = list(labels)

    def __getitem__(self, idx: int) -> tuple[T, int]:
        return self._get(self.files[idx]), self.labels[idx]

    @classmethod
    def from_dir(
        cls,
        root: str,
        label: int | Callable[[str], int],
        *,
        pattern: str | None = None,
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


class PathImageStackFolder(ImageStackFolderBase[ScalarType, T]):
    """Image stack folder with relpath."""

    root: str

    def __init__(self, files: Iterable[str], *, root: str, **kwargs):
        super().__init__(files, **kwargs)
        self.root = root

    def __getitem__(self, idx: int) -> tuple[T, str]:
        relpath = os.path.relpath(self.files[idx], self.root)
        return self._get(self.files[idx]), relpath

    @classmethod
    def from_dir(cls, root: str, *, pattern: str | None = None, **kwargs) -> Self:
        """
        Args:
            root: str
            pattern: Filter files by pattern.
            **kwargs: Pass to `cls.__init__`
        """

        return cls(cls.scan(root, pattern=pattern), root=root, **kwargs)


def truthly(*args, **kwargs) -> Literal[True]:  # pylint: disable=unused-argument
    return True
