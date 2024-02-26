"""Image stack pre-processing."""

import numpy as np
import numpy.typing as npt
from scipy.fftpack import fftn, fftshift, ifftn
from scipy.ndimage import gaussian_filter, minimum_filter

from swcgeom.transforms.base import Transform

__all__ = ["SGuoImPreProcess"]


class SGuoImPreProcess(Transform[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]):
    """Single-Neuron Image Enhancement.

    Implementation of the image enhancement method described in the paper:

    Shuxia Guo, Xuan Zhao, Shengdian Jiang, Liya Ding, Hanchuan Peng,
    Image enhancement to leverage the 3D morphological reconstruction
    of single-cell neurons, Bioinformatics, Volume 38, Issue 2,
    January 2022, Pages 503–512, https://doi.org/10.1093/bioinformatics/btab638
    """

    def __call__(self, x: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # TODO: support np.float32
        assert x.dtype == np.uint8, "Image must be in uint8 format"
        x = self.sigmoid_adjustment(x)
        x = self.subtract_min_along_z(x)
        x = self.bilateral_filter_3d(x)
        x = self.high_pass_fft(x)
        return x

    @staticmethod
    def sigmoid_adjustment(
        image: npt.NDArray[np.uint8], sigma: float = 3, percentile: float = 25
    ) -> npt.NDArray[np.uint8]:
        image_normalized = image / 255.0
        u = np.percentile(image_normalized, percentile)
        adjusted = 1 / (1 + np.exp(-sigma * (image_normalized - u)))
        adjusted_rescaled = (adjusted * 255).astype(np.uint8)
        return adjusted_rescaled

    @staticmethod
    def subtract_min_along_z(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        min_along_z = minimum_filter(
            image,
            size=(1, 1, image.shape[2], 1),
            mode="constant",
            cval=np.max(image).item(),
        )
        subtracted = image - min_along_z
        return subtracted

    @staticmethod
    def bilateral_filter_3d(
        image: npt.NDArray[np.uint8], spatial_sigma=(1, 1, 0.33), range_sigma=35
    ) -> npt.NDArray[np.uint8]:
        # initialize the output image
        filtered_image = np.zeros_like(image)

        spatial_gaussian = gaussian_filter(image, spatial_sigma)

        # traverse each pixel to perform bilateral filtering
        # TODO: optimization is needed
        for z in range(image.shape[2]):
            for y in range(image.shape[1]):
                for x in range(image.shape[0]):
                    value = image[x, y, z]
                    range_weight = np.exp(
                        -((image - value) ** 2) / (2 * range_sigma**2)
                    )
                    weights = spatial_gaussian * range_weight
                    filtered_image[x, y, z] = np.sum(image * weights) / np.sum(weights)

        return filtered_image

    @staticmethod
    def high_pass_fft(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # fft
        fft_image = fftn(image)
        fft_shifted = fftshift(fft_image)

        # create a high-pass filter
        h, w, d, _ = image.shape
        x, y, z = np.ogrid[:h, :w, :d]
        center = (h / 2, w / 2, d / 2)
        distance = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )
        # adjust this threshold to control the filtering strength
        high_pass_mask = distance > (d // 4)
        # apply the high-pass filter
        fft_shifted *= high_pass_mask

        # inverse fft
        fft_unshifted = np.fft.ifftshift(fft_shifted)
        filtered_image = np.real(ifftn(fft_unshifted))

        filtered_rescaled = np.clip(filtered_image, 0, 255).astype(np.uint8)
        return filtered_rescaled
