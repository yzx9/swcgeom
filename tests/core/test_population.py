import inspect
import io
import warnings

import pytest
from swcgeom.core.population import Population
from swcgeom.core.tree import Tree


class TestPopulationFromMultiRootsSWC:
    def test_single_root(self):
        SINGLE_ROOT_SWC = inspect.cleandoc("""
            # A comment line
            1 1 0 0 0 1 -1
            2 1 1 0 0 1 1
            3 1 0 1 0 1 1
            4 1 2 1 0 1 2
        """)
        swc_content = io.StringIO(SINGLE_ROOT_SWC)
        # Should warn that it's a single root file
        with pytest.warns(UserWarning, match="has only one root"):
            population = Population.from_multi_roots_swc(swc_content)

        assert isinstance(population, Population)
        assert len(population) == 1
        tree = population[0]
        assert isinstance(tree, Tree)
        assert tree.number_of_nodes() == 4
        assert tree.comments == ["A comment line"]

    MULTI_ROOT_SWC = inspect.cleandoc("""
        # Tree 1
        1 1 0 0 0 1 -1
        2 1 1 0 0 1 1
        # Tree 2
        10 1 10 0 0 1 -1
        11 1 11 0 0 1 10
        12 1 10 1 0 1 10
    """)

    def test_multi_root_reset_index(self):
        swc_content = io.StringIO(self.MULTI_ROOT_SWC)
        # Should not warn for multi-root files
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            population = Population.from_multi_roots_swc(
                swc_content, reset_index_per_subtree=True
            )

        assert isinstance(population, Population)
        assert len(population) == 2

        # Check tree 1 (original IDs 1, 2)
        tree1 = population[0] if population[0].number_of_nodes() == 2 else population[1]
        assert isinstance(tree1, Tree)
        assert tree1.number_of_nodes() == 2
        assert list(tree1.id()) == [0, 1]  # IDs reset
        assert list(tree1.pid()) == [-1, 0]  # PIDs reset
        assert tree1.comments == ["Tree 1", "Tree 2"]

        # Check tree 2 (original IDs 10, 11, 12)
        tree2 = population[0] if population[0].number_of_nodes() == 3 else population[1]
        assert isinstance(tree2, Tree)
        assert tree2.number_of_nodes() == 3
        assert list(tree2.id()) == [0, 1, 2]  # IDs reset
        assert list(tree2.pid()) == [-1, 0, 0]  # PIDs reset
        assert tree2.comments == ["Tree 1", "Tree 2"]

    def test_multi_root_no_reset_index(self):
        swc_content = io.StringIO(self.MULTI_ROOT_SWC)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            population = Population.from_multi_roots_swc(
                swc_content, reset_index_per_subtree=False
            )

        assert isinstance(population, Population)
        assert len(population) == 2

        # Check tree 1 (original IDs 1, 2)
        tree1 = population[0] if population[0].number_of_nodes() == 2 else population[1]
        assert isinstance(tree1, Tree)
        assert tree1.number_of_nodes() == 2
        assert list(tree1.id()) == [1, 2]  # IDs preserved
        assert list(tree1.pid()) == [-1, 1]  # PIDs preserved
        assert tree1.comments == ["Tree 1", "Tree 2"]

        # Check tree 2 (original IDs 10, 11, 12)
        tree2 = population[0] if population[0].number_of_nodes() == 3 else population[1]
        assert isinstance(tree2, Tree)
        assert tree2.number_of_nodes() == 3
        assert list(tree2.id()) == [10, 11, 12]  # IDs preserved
        assert list(tree2.pid()) == [-1, 10, 10]  # PIDs preserved
        assert tree2.comments == ["Tree 1", "Tree 2"]

    def test_empty_file(self):
        swc_content = io.StringIO("")  # Completely empty
        # Should warn about empty file
        with (
            pytest.warns(UserWarning, match="no trees in population"),
            pytest.warns(UserWarning, match="is empty or contains no valid nodes"),
        ):
            population = Population.from_multi_roots_swc(swc_content)

        assert isinstance(population, Population)
        assert len(population) == 0
        # The underlying read_swc_components returns empty comments for empty file
        # and Tree.from_data_frame gets None, so the trees (if any) would have no comments.

    def test_only_comments_file(self):
        EMPTY_SWC = inspect.cleandoc("""
            # Just comments
        """)
        swc_content = io.StringIO(EMPTY_SWC)
        # Should warn about empty file (after comments are stripped)
        with (
            pytest.warns(UserWarning, match="no trees in population"),
            pytest.warns(UserWarning, match="is empty or contains no valid nodes"),
        ):
            population = Population.from_multi_roots_swc(swc_content)

        assert isinstance(population, Population)
        assert len(population) == 0
        # Comments are extracted by read_swc_components, but no trees are created.

    def test_dangling_node(self):
        DANGLING_NODE_SWC = inspect.cleandoc("""
            # Root 1
            1 1 0 0 0 1 -1
            2 1 1 0 0 1 1
            # Root 2 (dangling node refers to non-existent parent 99)
            10 1 10 0 0 1 -1
            11 1 11 0 0 1 99 # Parent 99 does not exist
            12 1 10 1 0 1 10
        """)
        swc_content = io.StringIO(DANGLING_NODE_SWC)
        # read_swc_components should raise ValueError due to the dangling node
        # when trying to build the disjoint set union.
        with pytest.raises(Exception):
            Population.from_multi_roots_swc(swc_content)
