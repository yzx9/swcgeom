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


"""SWC format utils.

Notes
-----
This module provides a bunch of methods to manipulating swc files, they
are always trivial and unstabled, so we are NOT export it by default.
If you use the method here, please review the code more frequently, we
will try to flag all breaking changes but NO promises.
"""

from swcgeom.core.swc_utils.assembler import *
from swcgeom.core.swc_utils.base import *
from swcgeom.core.swc_utils.checker import *
from swcgeom.core.swc_utils.io import *
from swcgeom.core.swc_utils.normalizer import *
from swcgeom.core.swc_utils.subtree import *
