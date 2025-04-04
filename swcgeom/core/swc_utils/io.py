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


"""Read and write swc format."""

import re
import warnings
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from swcgeom.core.swc_utils.base import SWCNames, get_names
from swcgeom.core.swc_utils.checker import is_single_root
from swcgeom.core.swc_utils.normalizer import (
    link_roots_to_nearest_,
    mark_roots_as_somas_,
    reset_index_,
    sort_nodes_,
)
from swcgeom.utils import FileReader, PathOrIO

__all__ = ["read_swc", "to_swc"]


def read_swc(
    swc_file: PathOrIO,
    extra_cols: Iterable[str] | None = None,
    fix_roots: Literal["somas", "nearest", False] = False,
    sort_nodes: bool = False,
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
            After sorting the nodes, the index for each parent are always less than
            that of its children.
        reset_index: Reset node index to start with zero.
            DO NOT set to false if you are not sure what will happened.
        encoding: The name of the encoding used to decode the file.
            If is `detect`, we will try to detect the character encoding.

    Returns:
        df: ~pandas.DataFrame
        comments: List of string
    """
    names = get_names(names)
    df, comments = parse_swc(
        swc_file, names=names, extra_cols=extra_cols, encoding=encoding
    )

    # fix swc
    if fix_roots is not False and np.count_nonzero(df[names.pid] == -1) > 1:
        match fix_roots:
            case "somas":
                mark_roots_as_somas_(df)
            case "nearest":
                link_roots_to_nearest_(df)
            case _:
                raise ValueError(f"unknown fix type `{fix_roots}`")

    if sort_nodes:
        sort_nodes_(df)
    elif reset_index:
        reset_index_(df)

    # check swc
    if not is_single_root(df, names=names):
        warnings.warn(f"not a simple tree in `{swc_file}`")

    if (df[names.pid] == -1).argmax() != 0:
        warnings.warn(f"root is not the first node in `{swc_file}`")

    if (df[names.r] <= 0).any():
        warnings.warn(f"non-positive radius in `{swc_file}`")

    return df, comments


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
                if (match := re_swc.search(line)) is not None:
                    if flag and match.group(last_group):
                        warnings.warn(
                            f"some fields are ignored in row {i + 1} of `{fname}`"
                        )
                        flag = False

                    for i, trans in enumerate(transforms):
                        vals[i].append(trans(match.group(i + 1)))
                elif match := RE_COMMENT.match(line):
                    comment = line[len(match.group(0)) :].removesuffix("\n")
                    if not comment.startswith(ignored_comment):
                        comments.append(comment)
                elif not line.isspace():
                    raise ValueError(f"invalid row {i + 1} in `{fname}`")
        except UnicodeDecodeError as e:
            raise ValueError(
                "decode failed, try to enable auto detect `encoding='detect'`"
            ) from e

    df = pd.DataFrame.from_dict(dict(zip(keys, vals)))
    return df, comments
