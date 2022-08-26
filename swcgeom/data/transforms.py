from typing import Any, Generic, Iterable, TypeVar, cast, overload

T, K = TypeVar("T"), TypeVar("K")

T1, T2, T3 = TypeVar("T1"), TypeVar("T2"), TypeVar("T3")
T4, T5, T6 = TypeVar("T4"), TypeVar("T5"), TypeVar("T6")


class Transform(Generic[T, K]):
    def __call__(self, x: T) -> K:
        raise NotImplementedError()

    def get_name(self) -> str:
        raise NotImplementedError()


class Transforms(list[Transform[Any, Any]], Generic[T, K]):
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

    def apply(self, x: T) -> K:
        for transform in self:
            x = transform(x)

        return cast(K, x)

    def apply_batch(self, batch: Iterable[T]) -> Iterable[K]:
        return map(self.apply, batch)

    def get_names(self) -> Iterable[str]:
        return (transform.get_name() for transform in self)
