# SWCGEOM

[![Run tests](https://github.com/yzx9/swcgeom/actions/workflows/test.yml/badge.svg)](https://github.com/yzx9/swcgeom/actions/workflows/test.yml)
[![Release to GitHub](https://github.com/yzx9/swcgeom/actions/workflows/github-publish.yml/badge.svg)](https://github.com/yzx9/swcgeom/releases)
[![Release to PyPI](https://github.com/yzx9/swcgeom/actions/workflows/pypi-publish.yml/badge.svg)](https://pypi.org/project/swcgeom/)

A neuron geometry library for swc format.

## Usage

See examples for details.

## Development

```bash
# clone repo
git clone git@github.com:yzx9/swcgeom.git
cd swcgeom

# install dependencies
python -m pip install --upgrade pip
pip install build

# install editable version
pip install --editable .
```

Static analysis don't support import hook used in editable install for [PEP660](https://peps.python.org/pep-0660/) since upgrade to setuptools v64+, detail infomation at [setuptools#3518](https://github.com/pypa/setuptools/issues/3518), a workaround for vscode with pylance:

```json
{
    "python.analysis.extraPaths": ["/path/to/this/project"]
}
```

## LICENSE

This work is licensed under a <a rel="license" href="https://www.apache.org/licenses/">Apache-2.0</a>.

Copyright (c) 2022-present, Zexin Yuan
