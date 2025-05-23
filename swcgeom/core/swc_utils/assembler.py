
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Assemble lines to swc.

NOTE: This module is deprecated, please use `~.transforms.LinesToTree` instead.
"""

__all__ = ["assemble_lines", "try_assemble_lines"]


def assemble_lines(*args, **kwargs):
    """Assemble lines to tree.

    .. deprecated:: 0.15.0
        Use :meth:`~.transforms.LinesToTree` instead.
    """
    raise DeprecationWarning(
        "`assemble_lines` has been replaced by `~.transforms.LinesToTree` because it "
        "can be easy assemble with other tansformations.",
    )


def try_assemble_lines(*args, **kwargs):
    """Try assemble lines to tree.

    .. deprecated:: 0.15.0
        Use :meth:`~.transforms.LinesToTree` instead.
    """
    raise DeprecationWarning(
        "`try_assemble_lines` has been replaced by `~.transforms.LinesToTree` because "
        "it can be easy assemble with other tansformations.",
    )
