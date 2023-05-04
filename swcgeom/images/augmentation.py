"""Play augment in image stack."""

import random
from typing import List, Literal, Optional

import numpy as np
import numpy.typing as npt

__all__ = ["play_augment", "random_augmentations"]

NDArrf32 = npt.NDArray[np.float32]

# Augmentation = Literal[
#     "swap_xy",
#     "swap_xz",
#     "swap_yz",
#     "flip_x",
#     "flip_y",
#     "flip_z",
#     "rot90_xy",
#     "rot90_xz",
#     "rot90_yz",
# ]

IDENTITY = -1

augs = {
    # swaps
    "swap_xy": lambda x: np.swapaxes(x, 0, 1),
    "swap_xz": lambda x: np.swapaxes(x, 0, 2),
    "swap_yz": lambda x: np.swapaxes(x, 1, 2),
    # flips
    "flip_x": lambda x: np.flip(x, axis=[0]),
    "flip_y": lambda x: np.flip(x, axis=[1]),
    "flip_z": lambda x: np.flip(x, axis=[2]),
    # rotations
    "rot90_xy": lambda x: np.rot90(x, k=1, axes=(0, 1)),
    "rot90_xz": lambda x: np.rot90(x, k=1, axes=(0, 2)),
    "rot90_yz": lambda x: np.rot90(x, k=1, axes=(1, 2)),
}


class Augmentation:
    """Play augmentation."""

    def __init__(self, *, seed: int | None) -> None:
        self.seed = seed
        self.rand = random.Random(seed)

    def swapaxes(self, x, mode: Optional[Literal["xy", "xz", "yz"]] = None) -> NDArrf32:
        if mode is None:
            modes: List[Literal["xy", "xz", "yz"]] = ["xy", "xz", "yz"]
            mode = modes[self.rand.randint(0, 2)]

        match mode:
            case "xy":
                return np.swapaxes(x, 0, 1)
            case "xz":
                return np.swapaxes(x, 0, 2)
            case "yz":
                return np.swapaxes(x, 1, 2)
            case _:
                raise ValueError(f"invalid mode: {mode}")

    def flip(self, x, mode: Optional[Literal["xy", "xz", "yz"]] = None) -> NDArrf32:
        if mode is None:
            modes: List[Literal["xy", "xz", "yz"]] = ["xy", "xz", "yz"]
            mode = modes[random.randint(0, 2)]

        match mode:
            case "xy":
                return np.flip(x, axis=0)
            case "xz":
                return np.flip(x, axis=1)
            case "yz":
                return np.flip(x, axis=2)
            case _:
                raise ValueError(f"invalid mode: {mode}")


fns = list(augs.keys())


def play_augment(x: NDArrf32, method: Optional[Augmentation | int] = None) -> NDArrf32:
    """Play augment in x.

    Parameters
    ----------
    x : Array
        Array of shape (X, Y, Z, C)
    method : int or str, optional
        Augmentation method index / name. if not provided, a random
        augment will be apply.
    """

    if isinstance(method, str):
        key = method
    elif isinstance(method, int):
        key = fns[method]
    elif method is None:
        key = fns[random.randint(0, len(augs))]
    elif method == IDENTITY:
        return x
    else:
        raise ValueError("invalid augment method")

    return augs[key](x)


def random_augmentations(
    n: int, k: int, *, seed: int | None = None, include_identity: bool = True
) -> npt.NDArray[np.int64]:
    """Generate a sequence of augmentations.

    Parameters
    ----------
    n : int
        Size of image stacks.
    k : int
        Each image stack augmented to K image stack.
    seed : int | None, optional
        Random seed, forwarding to `random.Random`
    include_identity : bool, default `True`
        Include identity transform.

    Returns
    -------
    augmentations : List of (int, int)
        Sequence of length N * K, contains image index and augmentation
        method index.

    Examples
    --------
    ```python
    xs = os.listdir("path_to_imgs")
    augs = generate_random_augmentations(len(xs), 5)
    for i, j in range(augs):
        x = play_augment(read_imgs(os.path.join("path_to_imgs", xs[i])), j)
    ```
    """

    rand = random.Random(seed)
    seq = list(range(len(augs)))
    if include_identity:
        seq.append(IDENTITY)

    assert 0 < k < len(seq), "too large augment specify."

    augmentations = []
    for _ in range(n):
        rand.shuffle(seq)
        augmentations.extend(seq[:k])

    xs = np.stack([np.repeat(np.arange(n), k), augmentations])
    return xs
