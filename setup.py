# SPDX-FileCopyrightText: 2022-2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

import numpy
from setuptools import setup

setup(include_dirs=[numpy.get_include()])
