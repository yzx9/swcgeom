from typing import Any, Generic, Iterable, TypeVar, cast, overload

T, K = TypeVar("T"), TypeVar("K")

T1, T2, T3 = TypeVar("T1"), TypeVar("T2"), TypeVar("T3")
T4, T5, T6 = TypeVar("T4"), TypeVar("T5"), TypeVar("T6")


class Transform(Generic[T, K]):
    """An abstract class representing a :class:`Transform`.

    All transforms that represent a map from `T` to `K`. All subclasses should
    overwrite :meth:`apply`, supporting applying transform in `x`. Subclasses
    could also optionally overwrite :meth:`apply_batch`, which is expected to
    applying transform a batch of input, manual implementation usually gives
    better performance.
    """

    name: str

    def __init__(self, name: str) -> None:
        """"""
        super().__init__()
        self.name = name

    def __call__(self, x: T) -> K:
        """Apply transform."""
        return self.apply(x)

    def apply(self, x: T) -> K:
        """Apply transform."""
        raise NotImplementedError()

    def apply_batch(self, batch: Iterable[T]) -> Iterable[K]:
        """Apply transform in batch."""
        return map(self.apply, batch)

    def get_name(self) -> str:
        """Get transform name."""
        return self.name


class Transforms(list[Transform[Any, Any]], Generic[T, K]):
    """A simple typed wrapper for transforms."""

    # fmt:off
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
                       t7: Transform[T6, Any], /, *transforms: Transform[Any, Any]) -> None: ...
    # fmt:on

    def __init__(self, *transforms: Transform[Any, Any]) -> None:
        super().__init__(transforms)

    def __call__(self, x: T) -> K:
        """Apply transforms."""
        return self.apply(x)

    def apply(self, x: T) -> K:
        """Apply transforms."""
        for transform in self:
            x = transform.apply(x)

        return cast(K, x)

    def apply_batch(self, batch: Iterable[T]) -> Iterable[K]:
        """Apply transforms in batch."""
        for transform in self:
            batch = transform.apply_batch(batch)

        return cast(Iterable[K], batch)

    def get_name(self, separator: str = "_") -> str:
        """Get name of transforms.

        Parameters
        ----------
        separator : str, default `_`

        Returns
        -------
        name : str
            Name which join separator to the names.
        """
        return separator.join(self.get_names())

    def get_names(self) -> Iterable[str]:
        """Get names of transforms."""
        return (transform.get_name() for transform in self)
