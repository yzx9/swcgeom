"""Assemble a tree."""

from typing import List, Optional, Tuple

import pandas as pd

from ..core import Tree
from ..core.swc_utils import SWCNames
from ..core.swc_utils.assembler import assemble_lines_impl, try_assemble_lines_impl
from .base import Transform


class LinesToTree(Transform[List[pd.DataFrame], Tree]):
    """Assemble lines to swc."""

    def __init__(self, *, thre: float = 0.2, undirected: bool = True):
        """
        Parameters
        ----------
        thre : float, default `0.2`
            Connection threshold.
        undirected : bool, default `True`
            Both ends of a line can be considered connection point. If
            `False`, only the starting point.
        """
        super().__init__()
        self.thre = thre
        self.undirected = undirected

    def __call__(self, lines: List[pd.DataFrame], *, names: Optional[SWCNames] = None):
        return self.assemble(lines, names=names)

    def __repr__(self) -> str:
        return f"LinesToTree-thre-{self.thre}-{'undirected' if self.undirected else 'directed'}"

    def assemble(
        self, lines: List[pd.DataFrame], *, names: Optional[SWCNames] = None
    ) -> pd.DataFrame:
        """Assemble lines to a tree.

        Assemble all the lines into a set of subtrees, and then connect
        them.

        Parameters
        ----------
        lines : List of ~pd.DataFrame
            An array of tables containing a line, columns should follwing
            the swc.
        names : SWCNames, optional
            Forwarding to `self.try_assemble`.

        Returns
        -------
        tree : ~pd.DataFrame

        See Also
        --------
        self.try_assemble_lines
        """
        return assemble_lines_impl(
            lines, thre=self.thre, undirected=self.undirected, names=names
        )

    def try_assemble(
        self,
        lines: List[pd.DataFrame],
        *,
        id_offset: int = 0,
        sort_nodes: bool = True,
        names: Optional[SWCNames] = None,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Trying assemble lines to a tree.

        Treat the first line as a tree, find a line whose shortest distance
        between the tree and the line is less than threshold, merge it into
        the tree, repeat until there are no line to merge, return tree and
        the remaining lines.

        Parameters
        ----------
        lines : List of ~pd.DataFrame
            An array of tables containing a line, columns should follwing
            the swc.
        id_offset : int, default `0`
            The offset of the line node id.
        sort_nodes : bool, default `True`
            sort nodes of subtree.
        names : SWCNames, optional

        Returns
        -------
        tree : ~pandas.DataFrame
        remaining_lines : List of ~pandas.DataFrame
        """
        return try_assemble_lines_impl(
            lines,
            undirected=self.undirected,
            thre=self.thre,
            id_offset=id_offset,
            sort_nodes=sort_nodes,
            names=names,
        )
