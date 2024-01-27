"""Assemble lines to swc.

Notes
-----
This module is deprecated, please use `~.transforms.LinesToTree`
instead.
"""


__all__ = ["assemble_lines", "try_assemble_lines"]


def assemble_lines(*args, **kwargs):
    raise DeprecationWarning(
        "`assemble_lines` has been replaced by "
        "`~.transforms.LinesToTree` because it can be easy assemble "
        "with other tansformations.",
    )


def try_assemble_lines(*args, **kwargs):
    raise DeprecationWarning(
        "`try_assemble_lines` has been replaced by "
        "`~.transforms.LinesToTree` because it can be easy assemble "
        "with other tansformations.",
    )
