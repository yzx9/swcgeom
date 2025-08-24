# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Read and write swc format."""

import re
import warnings
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from swcgeom.core.swc_utils.base import SWCNames, get_names
from swcgeom.core.swc_utils.checker import get_num_of_roots, is_single_root
from swcgeom.core.swc_utils.normalizer import (
    link_roots_to_nearest_,
    mark_roots_as_somas_,
    reset_index_,
    sort_nodes_,
)
from swcgeom.utils import DisjointSetUnion, FileReader, PathOrIO

__all__ = ["read_swc", "read_swc_components", "to_swc"]


def read_swc(
    swc_file: PathOrIO,
    extra_cols: Iterable[str] | None = None,
    fix_roots: Literal["somas", "nearest", False] = False,
    sort_nodes: bool = True,
    reset_index: bool = True,
    *,
    encoding: Literal["detect"] | str = "utf-8",
    names: SWCNames | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read swc file.

    NOTE: the id should be consecutively incremented.

    Args:
        extra_cols: Read more cols in swc file.
        fix_roots: Fix multiple roots.
        sort_nodes: Sort the indices of neuron tree.
            After sorting the nodes, the index for each parent are always less than that of its children, default to
            True.
        reset_index: Reset node index to start with zero.
            DO NOT set to false if you are not sure what will happened, default to True.
        encoding: The name of the encoding used to decode the file.
            If is `detect`, we will try to detect the character encoding, default to "utf-8".

    Returns:
        df: ~pandas.DataFrame
        comments: List of string
    """
    names = get_names(names)
    df, comments = parse_swc(
        swc_file, names=names, extra_cols=extra_cols, encoding=encoding
    )
    if df.empty:
        raise ValueError(f"SWC file '{swc_file}' is empty or contains no valid nodes.")

    # fix swc
    if not is_single_root(df, names=names):
        match fix_roots:
            case "somas":
                mark_roots_as_somas_(df)
            case "nearest":
                link_roots_to_nearest_(df)
            case False:
                warnings.warn(f"not a simple tree in `{swc_file}`")
            case _:
                raise ValueError(f"unknown fix type `{fix_roots}`")

    # check swc
    if (df[names.pid] == -1).argmax() != 0:
        warnings.warn(f"root is not the first node in `{swc_file}`")

    if (df[names.r] <= 0).any():
        warnings.warn(f"non-positive radius in `{swc_file}`")

    # post processing
    if sort_nodes:
        sort_nodes_(df)
    elif reset_index:
        reset_index_(df)

    return df, comments


def read_swc_components(
    swc_file: PathOrIO,
    /,
    *,
    extra_cols: Iterable[str] | None = None,
    encoding: Literal["detect"] | str = "utf-8",
    names: SWCNames | None = None,
    reset_index_per_subtree: bool = True,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Read swc file, splitting multi-root files into separate DataFrames.

    If the SWC file contains multiple roots (disconnected components),
    each component is extracted into its own pandas DataFrame.

    Args:
        swc_file: Path to the SWC file.
        extra_cols: Read more cols in swc file.
        reset_index_per_subtree: Reset node index for each subtree
            to start with zero. Defaults to True.
        encoding: The name of the encoding used to decode the file.
            If 'detect', attempts to detect the character encoding.
        names: SWCNames configuration.

    Returns:
        dfs: A list of pandas DataFrames, each representing a
            connected component (potential tree) from the SWC file.
        comments: List of comment lines from the SWC file.
    """
    names = get_names(names)
    df, comments = parse_swc(
        swc_file, names=names, extra_cols=extra_cols, encoding=encoding
    )
    if df.empty:
        warnings.warn(f"SWC file '{swc_file}' is empty or contains no valid nodes.")
        return [], comments

    num_roots = get_num_of_roots(df, names=names)
    if num_roots == 0:
        warnings.warn(f"SWC file '{swc_file}' contains no root nodes (pid = -1).")
        return [], comments

    elif num_roots == 1:
        warnings.warn(
            f"SWC file '{swc_file}' has only one root. Consider using `read_swc` for single trees."
        )
        # Return the original DataFrame wrapped in a list
        return [df], comments

    # Multiple roots: Split into components
    sub_dfs = []
    num_nodes = len(df)
    dsu = DisjointSetUnion(num_nodes)

    # Map original node IDs to 0..N-1 indices for DSU
    id_to_idx = {node_id: i for i, node_id in enumerate(df[names.id])}
    for i, row in df.iterrows():
        parent_id = row[names.pid]
        if parent_id == -1:
            continue

        child_idx = i  # Use DataFrame index which is 0..N-1
        parent_idx = id_to_idx.get(parent_id)
        if parent_idx is None:
            warnings.warn(
                f"Parent ID {parent_id} for node ID {row[names.id]} not found "
                f"in '{swc_file}'. Treating node as root of a component."
            )
            continue

        # Ensure indices are valid before union (should always be if df is consistent)
        if not dsu.validate_node(child_idx) or not dsu.validate_node(parent_idx):
            # This case should ideally not happen with well-formed input
            warnings.warn(
                f"Internal error: Invalid node index for node id "
                f"{row[names.id]} or parent id {parent_id} in '{swc_file}'.",
                stacklevel=2,
            )
            continue

        dsu.union_sets(child_idx, parent_idx)

    # Group nodes by component representative index
    components: dict[int, list[int]] = {}
    for i in range(num_nodes):
        parent_repr = dsu.find_parent(i)
        if parent_repr not in components:
            components[parent_repr] = []
        components[parent_repr].append(i)  # Store original DataFrame indices (0..N-1)

    # Create a DataFrame for each component
    for component_indices in components.values():
        sub_df = df.iloc[component_indices].copy()

        if reset_index_per_subtree:
            # Remap IDs and PIDs for the subtree to be 0..M-1
            old_id_to_new_id = {
                old_id: new_id for new_id, old_id in enumerate(sub_df[names.id])
            }

            # Apply mapping, ensuring the root's PID becomes -1
            sub_df[names.id] = sub_df[names.id].map(old_id_to_new_id)
            sub_df[names.pid] = sub_df[names.pid].map(
                lambda old_pid: old_id_to_new_id.get(old_pid, -1)
            )
        # else: IDs remain as they were in the original file subset.

        sub_dfs.append(sub_df)

    return sub_dfs, comments


def to_swc(
    get_ndata: Callable[[str], npt.NDArray],
    *,
    extra_cols: Iterable[str] | None = None,
    id_offset: int = 1,
    comments: Iterable[str] | None = None,
    names: SWCNames | None = None,
) -> Iterable[str]:
    """Convert to swc format."""

    if comments is not None:
        for c in comments:
            if not c.isspace():
                yield f"# {c.lstrip()}\n"
            else:
                yield "#"

    names = get_names(names)
    cols = names.cols() + (list(extra_cols) if extra_cols is not None else [])
    yield f"# {' '.join(cols)}\n"

    def get_v(k: str, idx: int) -> str:
        vs = get_ndata(k)
        v = vs[idx]
        if np.issubdtype(vs.dtype, np.floating):
            return f"{v:.4f}"

        if k == names.id or (k == names.pid and v != -1):
            v += id_offset

        return str(v)

    for idx in get_ndata(names.id):
        yield " ".join(get_v(k, idx) for k in cols) + "\n"


RE_COMMENT = re.compile(r"^\s*#")
RE_FLOAT = r"([+-]?(?:\d+(?:[.]\d*)?(?:[eE][+-]?\d+)?|[.]\d+(?:[eE][+-]?\d+)?))"


def parse_swc(
    fname: PathOrIO,
    *,
    names: SWCNames,
    extra_cols: Iterable[str] | None = None,
    encoding: Literal["detect"] | str = "utf-8",
) -> tuple[pd.DataFrame, list[str]]:
    """Parse swc file.

    Args:
        encoding: The name of the encoding used to decode the file.
            If is `detect`, we will try to detect the character encoding.

    Returns:
        df: ~pandas.DataFrame
        comments: List of string
    """
    # pylint: disable=too-many-locals
    extras = list(extra_cols) if extra_cols else []

    keys = names.cols() + extras
    vals = [[] for _ in keys]
    transforms = [int, int, float, float, float, float, int] + [float for _ in extras]

    re_swc_cols = [
        r"([0-9]+)",  # id
        r"([0-9]+)",  # type
        RE_FLOAT,  # x
        RE_FLOAT,  # y
        RE_FLOAT,  # z
        RE_FLOAT,  # r
        r"(-?[0-9]+)",  # pid
    ] + [
        RE_FLOAT
        for _ in extras  # assert float
    ]

    re_swc_cols_str = r"\s+".join(re_swc_cols)
    # Leading spaces are allowed, as this is part of the data in
    # neuromorpho.org. More fields at the end is allowed, such as
    # reading eswc as swc, but with a warning.
    re_swc = re.compile(rf"^\s*{re_swc_cols_str}\s*([\s+-.0-9]*)$")

    last_group = 7 + len(extras) + 1
    ignored_comment = f"# {' '.join(names.cols())}"
    flag = True

    comments = []
    with FileReader(fname, encoding=encoding) as f:
        try:
            for i, line in enumerate(f):
                original_line = line  # Keep for error messages/comments
                line_content = original_line.split("#", 1)[0].strip()  # Process content part

                if not line_content:  # Skip empty lines or lines that become empty
                    # Handle full comment lines using original_line
                    if match := RE_COMMENT.match(original_line):
                        comment = original_line[len(match.group(0)) :].removesuffix("\n").strip()
                        if comment and not comment.startswith(ignored_comment):
                            comments.append(comment)
                    continue  # Move to next line

                if (match := re_swc.search(line_content)) is not None:
                    # Check for extra numerical fields captured by the last group
                    # Warn if the captured group exists and contains non-whitespace
                    if flag and match.group(last_group) and match.group(last_group).strip():
                        warnings.warn(
                            f"Extra fields detected and ignored in row {i + 1} of `{fname}`"
                        )
                        flag = False  # Only warn once

                    for j, trans in enumerate(transforms):
                        try:
                            vals[j].append(trans(match.group(j + 1)))
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid data format in row {i + 1}, column {j+1} ('{keys[j]}') in `{fname}`: {match.group(j + 1)}"
                            ) from e

                else:  # If re_swc didn't match the line_content
                    # It's not a valid SWC data line. We already handled empty/comment lines.
                    raise ValueError(
                        f"Invalid SWC data format in row {i + 1} in `{fname}`: {original_line.strip()}"
                    )

        except UnicodeDecodeError as e:
            raise ValueError(
                "decode failed, try to enable auto detect `encoding='detect'`"
            ) from e

    df = pd.DataFrame.from_dict(dict(zip(keys, vals)))
    return df, comments
