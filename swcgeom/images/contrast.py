# Copyright 2022-2025 Zexin Yuan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""The contrast of an image.

Notes
-----
This is expremental code, and the API is subject to change.
"""

from typing import Optional, overload

import numpy as np
import numpy.typing as npt

__all__ = ["contrast_std", "contrast_michelson", "contrast_rms", "contrast_weber"]

Array3D = npt.NDArray[np.float32]


@overload
def contrast_std(image: Array3D) -> float:
    """Get the std contrast of an image stack.

    Parameters
    ----------
    imgs : ndarray

    Returns
    -------
    contrast : float
    """
    ...


@overload
def contrast_std(image: Array3D, contrast: float) -> Array3D:
    """Adjust the contrast of an image stack.

    Parameters
    ----------
    imgs : ndarray
    constrast : float
        The contrast adjustment factor. 1.0 leaves the image unchanged.

    Returns
    -------
    imgs : ndarray
        The adjusted image.
    """
    ...


def contrast_std(image: Array3D, contrast: Optional[float] = None):
    if contrast is None:
        return np.std(image).item()
    else:
        return np.clip(contrast * image, 0, 1)


def contrast_michelson(image: Array3D) -> float:
    """Get the Michelson contrast of an image stack.

    Parameters
    ----------
    imgs : ndarray

    Returns
    -------
    contrast : float
    """

    vmax = np.max(image)
    vmin = np.min(image)
    return ((vmax - vmin) / (vmax + vmin)).item()


def contrast_rms(imgs: npt.NDArray[np.float32]) -> float:
    """Get the RMS contrast of an image stack.

    Parameters
    ----------
    imgs : ndarray

    Returns
    -------
    contrast : float
    """

    return np.sqrt(np.mean(imgs**2)).item()


def contrast_weber(imgs: Array3D, mask: npt.NDArray[np.bool_]) -> float:
    """Get the Weber contrast of an image stack.

    Parameters
    ----------
    imgs : ndarray
    mask : ndarray of bool
        The mask to segment the foreground and background. 1 for
        foreground, 0 for background.

    Returns
    -------
    contrast : float
    """

    l_foreground = np.mean(imgs, where=mask)
    l_background = np.mean(imgs, where=np.logical_not(mask))
    return ((l_foreground - l_background) / l_background).item()
