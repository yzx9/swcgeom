# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Debug helpers"""

import time
from functools import wraps

__all__ = ["func_timer"]


def func_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        print(f"[Function: {function.__name__} start...]")
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(f"[Function: {function.__name__} finished, spent time: {t1 - t0:.2f}s]")
        return result

    return function_timer
