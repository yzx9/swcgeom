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


"""Assemble lines to swc.

Notes
-----
This module is deprecated, please use `~.transforms.LinesToTree`
instead.
"""

__all__ = ["assemble_lines", "try_assemble_lines"]


def assemble_lines(*args, **kwargs):
    """Assemble lines to tree.

    .. deprecated:: 0.15.0
        Use :meth:`~.transforms.LinesToTree` instead.
    """

    raise DeprecationWarning(
        "`assemble_lines` has been replaced by "
        "`~.transforms.LinesToTree` because it can be easy assemble "
        "with other tansformations.",
    )


def try_assemble_lines(*args, **kwargs):
    """Try assemble lines to tree.

    .. deprecated:: 0.15.0
        Use :meth:`~.transforms.LinesToTree` instead.
    """

    raise DeprecationWarning(
        "`try_assemble_lines` has been replaced by "
        "`~.transforms.LinesToTree` because it can be easy assemble "
        "with other tansformations.",
    )
