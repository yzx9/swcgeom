"""SWC format utils.

Notes
-----
Methods in this module always receive `~pandas.DataFrame`, and assert
columns following naming defines in `~.core.swc`.
"""


from .assembler import *
from .base import *
from .checker import *
from .io import *
from .normalizer import *
from .subtree import *
