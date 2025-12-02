# SPDX-FileCopyrightText: 2022-2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy
from setuptools import Extension, setup


def has_cython():
    try:
        import Cython  # noqa

        return True
    except Exception:
        return False


use_cython = has_cython()


def maybe_ext(name, relpath_no_ext):
    # relpath_no_ext: e.g. "swcgeom/images/loaders/pbd"
    pyx = Path(relpath_no_ext + ".pyx")
    c = Path(relpath_no_ext + ".c")
    sources = [
        str(pyx if (use_cython and pyx.exists()) else (c if c.exists() else pyx))
    ]
    return Extension(name=name, sources=sources)


extensions = [
    maybe_ext("swcgeom.images.loaders.pbd", "swcgeom/images/loaders/pbd"),
    maybe_ext("swcgeom.images.loaders.raw", "swcgeom/images/loaders/raw"),
]


if use_cython:
    from Cython.Build import cythonize

    ext_modules = cythonize(extensions, language_level="3")
else:
    ext_modules = extensions

setup(include_dirs=[numpy.get_include()], ext_modules=ext_modules)
