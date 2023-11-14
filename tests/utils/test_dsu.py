"""Test DSU."""

from swcgeom.utils.dsu import DisjointSetUnion


def test_initialization():
    dsu = DisjointSetUnion(10)
    for i in range(10):
        assert dsu.find_parent(i) == i


def test_union_and_find():
    dsu = DisjointSetUnion(10)
    dsu.union_sets(0, 1)
    assert dsu.is_same_set(0, 1)
    assert not dsu.is_same_set(0, 2)

    dsu.union_sets(1, 2)
    assert dsu.is_same_set(0, 2)


def test_no_union():
    dsu = DisjointSetUnion(10)
    assert not dsu.is_same_set(0, 1)


def test_large_union():
    dsu = DisjointSetUnion(100)

    for i in range(99):
        dsu.union_sets(i, i + 1)

    for i in range(99):
        assert dsu.is_same_set(0, i)
