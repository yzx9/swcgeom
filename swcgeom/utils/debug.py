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
