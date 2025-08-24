import inspect
import io

from swcgeom.core.tree import Tree


class TestTreeFromSWC:
    def test_simple(self):
        SWC = inspect.cleandoc("""
            # A comment line
            1 1 0 0 0 1 -1
            2 1 1 0 0 1 1
            3 1 0 1 0 1 1
            4 1 2 1 0 1 2
        """)
        swc_content = io.StringIO(SWC)
        tree = Tree.from_swc(swc_content)

        assert isinstance(tree, Tree)
        assert tree.comments == ["A comment line"]
        assert tree.number_of_nodes() == 4
        assert len(tree.get_segments()) == 3
        assert len(tree.get_branches()) == 3
        assert len(tree.get_paths()) == 2
        assert len(tree.get_tips()) == 2
        assert len(tree.get_furcations()) == 1

    def test_line(self):
        SWC = inspect.cleandoc("""
            # A comment line
            1 1 0 0 0 1 -1
            2 1 1 0 0 1 1
            3 1 2 1 0 1 2
            4 1 3 1 0 1 3
        """)
        swc_content = io.StringIO(SWC)
        tree = Tree.from_swc(swc_content)

        assert isinstance(tree, Tree)
        assert tree.comments == ["A comment line"]
        assert tree.number_of_nodes() == 4
        assert len(tree.get_segments()) == 3
        assert len(tree.get_branches()) == 1
        assert len(tree.get_paths()) == 1
        assert len(tree.get_tips()) == 1
        assert len(tree.get_furcations()) == 0
