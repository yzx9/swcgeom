# -----------------------------------------------------------------------------
# This file is adapted from the v3d-py-helper project:
# https://github.com/SEU-ALLEN-codebase/v3d-py-helper
#
# Original license: MIT License
# Copyright (c) Zuohan Zhao
#
# Vaa3D in Python Made Easy
# Python library for Vaa3D functions.
#
# The original project is distributed via PyPI under the name `v3d-py-helper`,
# with its latest release (v0.4.1) not supporting Python 3.13.
#
# As of Python 3.13 (released on October 7, 2024), this package fails to build
# from source due to missing dependencies (e.g., libtiff), and no prebuilt wheels
# are available on PyPI for Python 3.13. An issue has been raised, but the project
# appears to be unmaintained at this time, and the author has not responded.
#
# To ensure continued compatibility and usability of Vaa3D features under Python 3.13+,
# we have copied and minimally adapted necessary source files into this project,
# preserving license and attribution in accordance with the MIT License.
#
# Please consult the original repository for full documentation:
# https://SEU-ALLEN-codebase.github.io/v3d-py-helper
#
# If the upstream project resumes maintenance and releases official support
# for Python 3.13+, this bundled version may be deprecated in favor of the
# canonical package.
# -----------------------------------------------------------------------------

import io
import os
import struct
import sys
import cython
cimport cython
import numpy as np
cimport numpy as np


DEF FORMAT_KEY_4 = b"raw_image_stack_by_hpeng"
DEF FORMAT_KEY_5 = b"raw5image_stack_by_hpeng"
DEF FORMAT_LEN = 24


cdef class Raw:
    """
    For vaa3d raw formats, allowing uint8, uint16 and float32 data types. The image dimension is presumed to be 4.

    As a raw format, it not only stores the image buffer, but also the image dimension sizes and its data type in the
    header. It also stores the endian of the buffer.

    This interface saves and loads a multi-dimension numpy array of either of the 3 data types. It also compares the endian of
    the file/numpy array and the machine to make it work properly. Based on the v3draw format key in the header, the
    array can be 4D or 5D.

    modified from v3d_external/v3d_main/basic_c_fun/stackutils.cpp

    by Zuohan Zhao

    2022/5/8
    """
    cdef:
        bint sz2byte

    def __init__(self, sz2byte = False):
        """

        :param sz2byte: set size array to be 2 byte (short int), for compatibility. Default as False.
        """
        self.sz2byte = sz2byte

    cpdef np.ndarray load(self, path: str | os.PathLike, int choose = -1):
        """
        :param path: input image path of v3draw.
        :param choose: choose a channel(4D) or stack(5D) to load, starting from 0, default as -1, meaning all.
        :return: a numpy array of 4D or 5D based on the format key.
        """
        cdef:
            short datatype
            list sz
            bytes endian_code_data
            str endian, dt
            char dim, header_sz, i
            bytes format_key
            long long bulk_sz, filesize

        filesize = os.path.getsize(path)
        assert filesize >= FORMAT_LEN, "File size too small, file might be corrupted"

        with open(path, "rb") as f:
            format_key = f.read(FORMAT_LEN)
            if format_key == FORMAT_KEY_4:
                dim = 4
            elif format_key == FORMAT_KEY_5:
                dim = 5
            else:
                raise RuntimeError("Format key isn't for v3draw")
            if self.sz2byte:
                header_sz = FORMAT_LEN + dim * 2 + 2 + 1
            else:
                header_sz = FORMAT_LEN + dim * 4 + 2 + 1
            assert filesize >= header_sz, "File size too small, file might be corrupted"
            endian_code_data = f.read(1)
            if endian_code_data == b'B':
                endian = '>'
            elif endian_code_data == b'L':
                endian = '<'
            else:
                raise RuntimeError('Endian code should be either B/L')
            datatype = struct.unpack(f'{endian}h', f.read(2))[0]
            if datatype == 1:
                dt = 'u1'
            elif datatype == 2:
                dt = 'u2'
            elif datatype == 4:
                dt = 'f4'
            else:
                raise RuntimeError('v3draw data type can only be 1/2/4')
            if self.sz2byte:
                sz = list(struct.unpack(f'{endian}{dim}h', f.read(dim * 2)))
            else:
                sz = list(struct.unpack(f'{endian}{dim}i', f.read(dim * 4)))
            bulk_sz = 1
            for i in range(dim - 1):
                bulk_sz *= sz[i]
            assert bulk_sz * sz[-1] * datatype + header_sz == filesize, "file size doesn't match with the image"
            if choose < 0:
                return np.frombuffer(f.read(), endian + dt).reshape(sz[::-1])
            else:
                assert choose < sz[-1], "Choose index exceeding the range"
                f.seek(bulk_sz * datatype * choose, 1)
                img = np.frombuffer(f.read(bulk_sz * datatype), endian + dt)
                return img.reshape(sz[-2::-1])

    cpdef void save(self, path: str | os.PathLike, np.ndarray img):
        """
        :param path: output image path of v3draw.
        :param img: the image array to save. 
        """
        assert img.ndim == 4, "The image has to be 4D"
        assert img.dtype in [np.uint8, np.uint16], "The pixel type has to be uint8 or uint16"
        cdef:
            short datatype
            list sz
            bytes endian_code_data, format
            str endian, bo = img.dtype.byteorder
            bytes header
            char dim = img.ndim

        if dim == 4:
            format = FORMAT_KEY_4
        elif dim == 5:
            format = FORMAT_KEY_5
        else:
            raise RuntimeError("Dimension not supported by v3draw")

        sz = [img.shape[i] for i in range(dim)]
        sz.extend([1] * (dim - len(sz)))
        sz.reverse()

        if bo == '>' or bo == '=' and sys.byteorder == 'big':
            endian_code_data = b'B'
            endian = '>'
        else:
            endian_code_data = b'L'
            endian = '<'

        if img.dtype == np.uint8:
            datatype = 1
        elif img.dtype == np.uint16:
            datatype = 2
        elif img.dtype == np.float32:
            datatype = 4
        else:
            raise RuntimeError("numpy data type not supported by v3draw")

        header = struct.pack(f'{endian}{FORMAT_LEN}sch4{"h" if self.sz2byte else "i"}',
                             format, endian_code_data, datatype, *sz)

        with open(path, 'wb') as f:
            f.write(header)
            f.write(img.tobytes())
