# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Disjoint Set Union Impl."""

__all__ = ["DisjointSetUnion"]


class DisjointSetUnion:
    """Disjoint Set Union.

    DSU with path compression and union by rank.

    >>> dsu = DisjointSetUnion(3)
    >>> dsu.is_same_set(0, 1)
    False
    >>> dsu.union_sets(0, 1)
    >>> dsu.is_same_set(0, 1)
    True
    >>> dsu.is_same_set(0, 2)
    False
    >>> dsu.union_sets(1, 2)
    >>> dsu.is_same_set(0, 2)
    True
    """

    def __init__(self, node_number: int):
        self.element_parent = [i for i in range(node_number)]
        self.rank = [0 for _ in range(node_number)]

    def find_parent(self, node_id: int) -> int:
        if node_id != self.element_parent[node_id]:
            self.element_parent[node_id] = self.find_parent(
                self.element_parent[node_id]
            )
        return self.element_parent[node_id]

    def union_sets(self, node_a: int, node_b: int) -> None:
        assert self.validate_node(node_a) and self.validate_node(node_b)

        root_a = self.find_parent(node_a)
        root_b = self.find_parent(node_b)
        if root_a != root_b:
            # union by rank
            if self.rank[root_a] < self.rank[root_b]:
                self.element_parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.element_parent[root_b] = root_a
            else:
                self.element_parent[root_b] = root_a
                self.rank[root_a] += 1

    def is_same_set(self, node_a: int, node_b: int) -> bool:
        return self.find_parent(node_a) == self.find_parent(node_b)

    def validate_node(self, node_id: int) -> bool:
        return 0 <= node_id < len(self.element_parent)
