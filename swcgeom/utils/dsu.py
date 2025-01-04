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


"""Disjoint Set Union Impl."""

__all__ = ["DisjointSetUnion"]


class DisjointSetUnion:
    """Disjoint Set Union.

    DSU with path compression and union by rank.
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
