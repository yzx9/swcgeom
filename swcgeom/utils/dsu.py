"""Disjoint Set Union Impl."""

__all__ = ["DisjointSetUnion"]


class DisjointSetUnion:
    """Disjoint Set Union."""
    def __init__(self, node_number: int):
        self.element_parent = {}
        for a in range(node_number):
            self.element_parent[a] = a

    def is_same_set(self, node_a: int, node_b: int) -> bool:
        return self.find_parent(node_a) == self.find_parent(node_b)

    def union_sets(self, node_a: int, node_b: int) -> None:
        node_a_parent = self.find_parent(node_a)
        node_b_parent = self.find_parent(node_b)
        if not node_a_parent == node_b_parent:
            self.element_parent[node_a_parent] = node_b_parent

    def find_parent(self, node_id: int) -> int:
        if self.element_parent[node_id] == node_id:
            return node_id
        else:
            return self.find_parent(self.element_parent[node_id])
