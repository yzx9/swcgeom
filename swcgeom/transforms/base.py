# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Transformation in tree."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, overload

from typing_extensions import override

__all__ = ["Transform", "Transforms", "Identity"]

T = TypeVar("T")
K = TypeVar("K")

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
T6 = TypeVar("T6")


class Transform(ABC, Generic[T, K]):
    r"""An abstract class representing a :class:`Transform`.

    All transforms that represent a map from `T` to `K`.
    """

    @abstractmethod
    def __call__(self, x: T) -> K:
        """Apply transform.

        NOTE: All subclasses should overwrite :meth:`__call__`, supporting
        applying transform in `x`.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        repr_ = self.extra_repr()
        return f"{classname}({repr_})"

    def extra_repr(self) -> str:
        """Provides a human-friendly representation of the module.

        This method extends the basic string representation provided by
        `__repr__` method. It is designed to display additional details
        about the module's parameters or its specific configuration,
        which can be particularly useful for debugging and model
        architecture introspection.

        >>> class Foo(Transform[T, K]):
        ...     def __init__(self, my_parameter: int = 1):
        ...         self.my_parameter = my_parameter
        ...
        ...     def extra_repr(self) -> str:
        ...         return f"my_parameter={self.my_parameter}"

        NOTE: This method should be overridden in custom modules to provide
        specific details relevant to the module's functionality and
        configuration.
        """
        return ""


class Transforms(Transform[T, K]):
    """A simple typed wrapper for transforms."""

    transforms: list[Transform[Any, Any]]

    # fmt: off
    @overload
    def __init__(self, t1: Transform[T, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, T4],
                       t5: Transform[T4, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, T4],
                       t5: Transform[T4, T5], t6: Transform[T5, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, T4],
                       t5: Transform[T4, T5], t6: Transform[T5, T6],
                       t7: Transform[T6, K], /) -> None: ...
    @overload
    def __init__(self, t1: Transform[T, T1],  t2: Transform[T1, T2],
                       t3: Transform[T2, T3], t4: Transform[T3, T4],
                       t5: Transform[T4, T5], t6: Transform[T5, T6],
                       t7: Transform[T6, Any], /, *transforms: Transform[Any, K]) -> None: ...
    # fmt: on
    def __init__(self, *transforms: Transform[Any, Any]) -> None:
        trans = []
        for t in transforms:
            if isinstance(t, Transforms):
                trans.extend(t.transforms)
            else:
                trans.append(t)
        self.transforms = trans

    @override
    def __call__(self, x: T) -> K:
        """Apply transforms."""
        for transform in self.transforms:
            x = transform(x)

        return x  # type: ignore

    def __getitem__(self, idx: int) -> Transform[Any, Any]:
        return self.transforms[idx]

    def __len__(self) -> int:
        return len(self.transforms)

    @override
    def extra_repr(self) -> str:
        return ", ".join([str(transform) for transform in self])


class Identity(Transform[T, T]):
    """Resurn input as-is."""

    @override
    def __call__(self, x: T) -> T:
        return x
