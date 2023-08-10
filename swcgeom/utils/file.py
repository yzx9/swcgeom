import warnings
from io import TextIOWrapper
from typing import Literal

import chardet

__all__ = ["FileReader"]


class FileReader:
    def __init__(
        self,
        fname: str,
        *,
        encoding: Literal["detect"] | str = "utf-8",
        low_confidence: float = 0.9,
        **kwargs,
    ) -> None:
        """Read file.

        Parameters
        ----------
        fname : str
        encoding : str | 'detect', default `utf-8`
            The name of the encoding used to decode the file. If is
            `detect`, we will try to detect the character encoding.
        low_confidence : float, default to 0.9
            Used for detect character endocing, raising warning when
            parsing with low confidence.
        """
        self.fname = fname
        if encoding == "detect":
            encoding = detect_encoding(fname, low_confidence=low_confidence)
        self.encoding = encoding
        self.kwargs = kwargs

    def __enter__(self) -> TextIOWrapper:
        self.f = open(self.fname, "r", encoding=self.encoding, **self.kwargs)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.f:
            self.f.close()
        return True


def detect_encoding(fname: str, *, low_confidence: float = 0.9) -> str:
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
