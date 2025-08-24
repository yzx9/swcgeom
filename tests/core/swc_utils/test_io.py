import inspect
import io
import warnings

import pandas as pd
import pytest
from swcgeom.core.swc_utils.io import read_swc, read_swc_components


class TestReadSWC:
    def test_line(self):
        LINE_SWC = inspect.cleandoc("""
            # A comment line
            1 1 0 0 0 1 -1
            2 1 1 0 0 1 1
            3 1 2 0 0 1 2
            4 1 3 0 0 1 3
        """)
        swc_content = io.StringIO(LINE_SWC)
        df, comments = read_swc(swc_content)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert comments == ["A comment line"]


class TestReadSWCComponents:
    def test_single_root(self):
        SINGLE_ROOT_SWC = inspect.cleandoc("""
            # A comment line
            1 1 0 0 0 1 -1
            2 1 1 0 0 1 1
            3 1 0 1 0 1 1
            4 1 2 1 0 1 2
        """)
        swc_content = io.StringIO(SINGLE_ROOT_SWC)
        with pytest.warns(UserWarning, match="has only one root"):
            dfs, comments = read_swc_components(swc_content)

        assert isinstance(dfs, list)
        assert len(dfs) == 1
        assert isinstance(dfs[0], pd.DataFrame)
        assert len(dfs[0]) == 4
        assert comments == ["A comment line"]

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
            dfs, comments = read_swc_components(
                swc_content, reset_index_per_subtree=True
            )

        assert isinstance(dfs, list)
        assert len(dfs) == 2
        assert comments == ["Tree 1", "Tree 2"]

        # Check df 1 (original IDs 1, 2)
        df1 = dfs[0] if len(dfs[0]) == 2 else dfs[1]
        assert isinstance(df1, pd.DataFrame)
        assert len(df1) == 2
        assert list(df1["id"]) == [0, 1]  # IDs reset
        assert list(df1["pid"]) == [-1, 0]  # PIDs reset

        # Check df 2 (original IDs 10, 11, 12)
        df2 = dfs[0] if len(dfs[0]) == 3 else dfs[1]
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == 3
        assert list(df2["id"]) == [0, 1, 2]  # IDs reset
        assert list(df2["pid"]) == [-1, 0, 0]  # PIDs reset

    def test_multi_root_no_reset_index(self):
        swc_content = io.StringIO(self.MULTI_ROOT_SWC)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dfs, comments = read_swc_components(
                swc_content, reset_index_per_subtree=False
            )

        assert isinstance(dfs, list)
        assert len(dfs) == 2
        assert comments == ["Tree 1", "Tree 2"]

        # Check df 1 (original IDs 1, 2)
        df1 = dfs[0] if len(dfs[0]) == 2 else dfs[1]
        assert isinstance(df1, pd.DataFrame)
        assert len(df1) == 2
        assert list(df1["id"]) == [1, 2]  # IDs preserved
        assert list(df1["pid"]) == [-1, 1]  # PIDs preserved

        # Check df 2 (original IDs 10, 11, 12)
        df2 = dfs[0] if len(dfs[0]) == 3 else dfs[1]
        assert isinstance(df2, pd.DataFrame)
        assert len(df2) == 3
        assert list(df2["id"]) == [10, 11, 12]  # IDs preserved
        assert list(df2["pid"]) == [-1, 10, 10]  # PIDs preserved

    def test_empty_file(self):
        EMPTY_SWC = inspect.cleandoc("""
            # Just comments
        """)
        swc_content = io.StringIO(EMPTY_SWC)
        with pytest.warns(UserWarning, match="is empty or contains no valid nodes"):
            dfs, comments = read_swc_components(swc_content)

        assert isinstance(dfs, list)
        assert len(dfs) == 0
        assert comments == ["Just comments"]

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
        # The warning about the dangling node should be emitted
        with pytest.raises(Exception):  # FIXME: We should handle this better
            read_swc_components(swc_content)
