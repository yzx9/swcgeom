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


"""Test DSU."""

from swcgeom.utils.dsu import DisjointSetUnion


class TestDSU:
    def test_initialization(self):
        dsu = DisjointSetUnion(10)
        for i in range(10):
            assert dsu.find_parent(i) == i

    def test_union_and_find(self):
        dsu = DisjointSetUnion(10)
        dsu.union_sets(0, 1)
        assert dsu.is_same_set(0, 1)
        assert not dsu.is_same_set(0, 2)

        dsu.union_sets(1, 2)
        assert dsu.is_same_set(0, 2)

    def test_no_union(self):
        dsu = DisjointSetUnion(10)
        assert not dsu.is_same_set(0, 1)

    def test_large_union(self):
        dsu = DisjointSetUnion(100)

        for i in range(99):
            dsu.union_sets(i, i + 1)

        for i in range(99):
            assert dsu.is_same_set(0, i)
