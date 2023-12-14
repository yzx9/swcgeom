"""File related utils.

Notes
-----
If character coding is enabled, all denpendencies need to be installed,
try:

```sh
pip install swcgeom[all]
```
"""

import warnings
from io import BytesIO, TextIOBase, TextIOWrapper
from typing import Literal

__all__ = ["FileReader", "PathOrIO"]

PathOrIO = int | str | bytes | BytesIO | TextIOBase


class FileReader:
    def __init__(
        self,
        fname: PathOrIO,
        *,
        encoding: Literal["detect"] | str = "utf-8",
        low_confidence: float = 0.9,
        **kwargs,
    ) -> None:
        """Read file.

        Parameters
        ----------
        fname : PathOrIO
        encoding : str | 'detect', default `utf-8`
            The name of the encoding used to decode the file. If is
            `detect`, we will try to detect the character encoding.
        low_confidence : float, default to 0.9
            Used for detect character endocing, raising warning when
            parsing with low confidence.
        """
        # TODO: support StringIO
        self.fname, self.fb, self.f = "", None, None
        if isinstance(fname, TextIOBase):
            self.f = fname
            encoding = fname.encoding  # skip detect
        elif isinstance(fname, BytesIO):
            self.fb = fname
        else:
            self.fname = fname

        if encoding == "detect":
            encoding = detect_encoding(fname, low_confidence=low_confidence)
        self.encoding = encoding
        self.kwargs = kwargs

    def __enter__(self) -> TextIOBase:
        if isinstance(self.fb, BytesIO):
            self.f = TextIOWrapper(self.fb, encoding=self.encoding)
        elif self.f is None:
            self.f = open(self.fname, "r", encoding=self.encoding, **self.kwargs)

        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.f:
            self.f.close()
        return True


def detect_encoding(fname: PathOrIO, *, low_confidence: float = 0.9) -> str:
    import chardet

    if isinstance(fname, TextIOBase):
        return fname.encoding
    elif isinstance(fname, BytesIO):
        data = fname.read()
        fname.seek(0, 0)
    else:
        with open(fname, "rb") as f:
            data = f.read()

    result = chardet.detect(data)
    encoding = result["encoding"] or "utf-8"
    if result["confidence"] < low_confidence:
        warnings.warn(
            f"parse as `{encoding}` with low confidence "
            f"{result['confidence']} in `{fname}`"
        )
    return encoding
