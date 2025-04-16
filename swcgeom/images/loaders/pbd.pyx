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

import struct
import os
import cython
cimport cython
import numpy as np
cimport numpy as np
import sys

from cpython.bytearray cimport PyByteArray_AsString
from libc.stdio cimport FILE, fopen, fread, fclose, fwrite


DEF FORMAT_KEY = b"v3d_volume_pkbitdf_encod"
DEF HEADER_SIZE = 43
DEF COMPRESSION_ENLARGEMENT = 2
DEF LITTLE = b'L'
DEF REPEAT_MAX_LEN  = 255 - 222

cdef unsigned char[3] MAX_LEN = [79 - 31, 182 - 79, 222 - 182]
cdef double[3] MAX_EFF = [16. / 3., 16. / 4., 16. / 5.]
cdef char[3][2] ran = [[-3, 4], [-7, 8], [-15, 16]]
cdef unsigned char[3] shift_bits = [3, 4, 5]
cdef unsigned char[3] gap = [31, 79, 182]
cdef unsigned char[3] mask = [0b00000111, 0b0001111, 0b00011111]


cdef class PBD:
    """
    Supporting most biological image LOSSLESS compression with sound SNR, where blocks of the signal are close in intensity.
    It supports 16bit and 8bit compression, can compress up to 25% or even lower. Note that the 16bit used by other sources
    is incomplete and can only reach 50%. This package gives you a choice of a full blood one that can make 25%, but the
    full blood output might not be loaded by other programs currently.

    The compression/decompression are optimized and can even be faster than Vaa3D.

    modified from v3d_external/v3d_main/neuron_annotator/utility/ImageLoaderBasic.cpp

    by Zuohan Zhao, Southeast University

    2022/6/23
    """
    cdef:
        long long total_read_bytes, compression_pos, decompression_pos, read_step_size_bytes
        bytearray compression_buffer, decompression_buffer
        bint endian_switch, pbd16_full_blood
        bytes endian_sys

    def __init__(self, pbd16_full_blood=True, read_step_size_bytes = 1024 * 20000):
        """
        :param pbd16_full_blood: Turn off or on to allow the full blood saving of 16bit image loading. Note
         other programs may not be able to load it. Default is on. Default as turned on.
        :param read_step_size_bytes: Adjust the number of bytes for each time of buffer loading, default as 20000KB.
        """
        self.endian_sys = sys.byteorder[0].upper().encode('ascii')
        self.endian_switch = False
        self.decompression_pos = self.compression_pos = self.total_read_bytes = 0
        self.compression_buffer = self.decompression_buffer = bytearray()
        self.pbd16_full_blood = pbd16_full_blood
        self.read_step_size_bytes = read_step_size_bytes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long long decompress_pbd8(self, long long look_ahead):
        cdef:
            unsigned char* decomp = <unsigned char*>PyByteArray_AsString(self.decompression_buffer)
            unsigned char* comp = <unsigned char*>PyByteArray_AsString(self.compression_buffer)
            long long cp = self.compression_pos, dp = self.decompression_pos
            unsigned char count, shift, carry
            char delta
        while cp < look_ahead:
            count = comp[cp]
            cp += 1
            if count < 33:
                count += 1
                for shift in range(count):
                    decomp[dp + shift] = comp[cp + shift]
                cp += count
                dp += count
            elif count < 128:
                count -= 32
                shift = 0
                while count > 0:
                    if shift == 0:
                        carry = comp[cp]
                        cp += 1
                    delta = (carry & 0b00000011 << shift) >> shift
                    if delta == 3:
                        delta = -1
                    decomp[dp] = decomp[dp - 1] + delta
                    dp += 1
                    count -= 1
                    shift = shift + 2 & 7
            else:
                count -= 127
                for shift in range(count):
                    decomp[dp + shift] = comp[cp]
                dp += count
                cp += 1
        return dp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long long decompress_pbd16(self, long long look_ahead):
        cdef:
            unsigned char * decomp = <unsigned char *> PyByteArray_AsString(self.decompression_buffer)
            unsigned char * comp = <unsigned char *> PyByteArray_AsString(self.compression_buffer)
            long long cp = self.compression_pos, dp = self.decompression_pos
            unsigned char count, i
            char delta, shift
            unsigned short carry
            unsigned short* ptr

        while cp < look_ahead:
            count = comp[cp]
            cp += 1
            if count < 32:
                count = count + 1 << 1
                if self.endian_switch:
                    for i in range(0, count, 2):
                        decomp[dp + i] = comp[cp + i + 1]
                        decomp[dp + i + 1] = comp[cp + i]
                else:
                    for i in range(0, count, 2):
                        decomp[dp + i] = comp[cp + i]
                        decomp[dp + i + 1] = comp[cp + i + 1]
                cp += count
                dp += count
                continue
            elif count < 80:
                i = 0
            elif count < 183:
                i = 1
            elif count < 223:
                i = 2
            else:
                count = count - 222 << 1
                if self.endian_switch:
                    for i in range(0, count, 2):
                        decomp[dp + i] = comp[cp + 1]
                        decomp[dp + i + 1] = comp[cp]
                else:
                    for i in range(0, count, 2):
                        decomp[dp + i] = comp[cp]
                        decomp[dp + i + 1] = comp[cp + 1]
                dp += count
                cp += 2
                continue
            count -= gap[i]
            shift = 0
            ptr = <unsigned short *> &decomp[dp]
            while count > 0:
                shift -= shift_bits[i]
                if shift < 0:
                    carry = carry << 8 | comp[cp]
                    cp += 1
                    shift += 8
                delta = carry >> shift & mask[i]
                if delta > ran[i][1]:
                    delta = ran[i][1] - delta
                ptr[0] = ptr[-1] + delta
                ptr += 1
                dp += 2
                count -= 1
        return dp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_compression_buffer8(self):
        cdef:
            long long look_ahead = self.compression_pos
            unsigned char lav, compressed_diff_entries
        while look_ahead < self.total_read_bytes:
            lav = self.compression_buffer[look_ahead]
            if lav < 33:
                if look_ahead + lav + 1 < self.total_read_bytes:
                    look_ahead += lav + 2
                else:
                    break
            elif lav < 128:
                compressed_diff_entries = (lav - 33) // 4 + 1
                if look_ahead + compressed_diff_entries < self.total_read_bytes:
                    look_ahead += compressed_diff_entries + 1
                else:
                    break
            else:
                if look_ahead + 1 < self.total_read_bytes:
                    look_ahead += 2
                else:
                    break
        self.decompression_pos = self.decompress_pbd8(look_ahead)
        self.compression_pos = look_ahead

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_compression_buffer16(self):
        cdef:
            long long look_ahead = self.compression_pos
            unsigned char lav, compressed_diff_bytes
        while look_ahead < self.total_read_bytes:
            lav = self.compression_buffer[look_ahead]
            if lav < 32:
                if look_ahead + (lav + 1) * 2 < self.total_read_bytes:
                    look_ahead += (lav + 1) * 2 + 1
                else:
                    break
            elif lav < 80:
                compressed_diff_bytes = ((lav - 31) * 3 - 1) // 8 + 1
                if look_ahead + compressed_diff_bytes < self.total_read_bytes:
                    look_ahead += compressed_diff_bytes + 1
                else:
                    break
            elif lav < 183:
                compressed_diff_bytes = ((lav - 79) * 4 - 1) // 8 + 1
                if look_ahead + compressed_diff_bytes < self.total_read_bytes:
                    look_ahead += compressed_diff_bytes + 1
                else:
                    break
            elif lav < 223:
                compressed_diff_bytes = ((lav - 182) * 5 - 1) // 8 + 1
                if look_ahead + compressed_diff_bytes < self.total_read_bytes:
                    look_ahead += compressed_diff_bytes + 1
                else:
                    break
            else:
                if look_ahead + 2 < self.total_read_bytes:
                    look_ahead += 3
                else:
                    break
        self.decompression_pos = self.decompress_pbd16(look_ahead)
        self.compression_pos = look_ahead

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray load(self, path: str | os.PathLike):
        """
        :param path: output image path of v3dpbd.
        :return: a 4D numpy array of either uint8 or uint16.
        """
        file_size = os.path.getsize(path)
        assert file_size >= HEADER_SIZE, "File size smaller than header size."
        cdef:
            short datatype
            long long current_read_bytes, channel_len, remaining_bytes
            const unsigned char[:] p = str(path).encode('utf-8')
            FILE* f = fopen(<const char*>&p[0], <char*>'rb')
        if f is NULL:
            raise Exception("Fail to open file for reading.")
        try:
            header = bytearray(HEADER_SIZE)
            fread(PyByteArray_AsString(header), HEADER_SIZE, 1, f)
            assert header.find(FORMAT_KEY) == 0, "Format key loading failed."
            header = header[len(FORMAT_KEY):]
            endian = '<' if header[:1] == LITTLE else '>'
            self.endian_switch = header[:1] != self.endian_sys
            datatype = struct.unpack(f'{endian}h', header[1:3])[0]
            assert datatype in [1, 2], "Datatype can only be 1 or 2."
            sz = struct.unpack(f'{endian}iiii', header[3:])
            channel_len = sz[0] * sz[1] * sz[2]
            remaining_bytes = file_size - HEADER_SIZE
            self.total_read_bytes = 0
            self.compression_buffer = bytearray(remaining_bytes)
            self.decompression_buffer = bytearray(channel_len * sz[3] * datatype)
            self.compression_pos = self.decompression_pos = 0
            while remaining_bytes > 0:
                current_read_bytes = min(remaining_bytes, self.read_step_size_bytes,
                                         (self.total_read_bytes // channel_len + 1) *
                                         channel_len - self.total_read_bytes)
                fread(PyByteArray_AsString(self.compression_buffer) + self.total_read_bytes, current_read_bytes, 1, f)
                self.total_read_bytes += current_read_bytes
                remaining_bytes -= current_read_bytes
                if datatype == 1:
                    self.update_compression_buffer8()
                elif datatype == 2:
                    self.update_compression_buffer16()
                else:
                    raise Exception("Invalid datatype")
            return np.frombuffer(self.decompression_buffer, f'{endian}u{datatype}').reshape(sz[::-1])
        finally:
            fclose(f)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long long compress_pbd8(self):
        cdef:
            unsigned char cur_val, prior_val, retest
            unsigned char[96] dbuffer
            unsigned char * decomp = <unsigned char *> PyByteArray_AsString(self.decompression_buffer)
            unsigned char * comp = <unsigned char *> PyByteArray_AsString(self.compression_buffer)
            short delta
            long long active_literal_index = -1, cur_dp, cp = 0, dp = 0
            double re_efficiency, df_efficiency
        assert self.decompression_pos > 0, "The buffer to save is empty."
        while dp < self.decompression_pos:
            if cp >= self.compression_pos:
                raise Exception("compression running out of space, try enlarging the compression buffer.")
            retest = 1
            cur_val = decomp[dp]
            cur_dp = dp + 1
            while cur_dp < self.decompression_pos and retest < 128 and decomp[cur_dp] == cur_val:
                retest += 1
                cur_dp += 1
            re_efficiency = retest / 2.

            if re_efficiency < 4:
                df_efficiency = 0.
                cur_dp = dp
                if dp > 0:
                    prior_val = decomp[dp - 1]
                    for cur_dp in range(dp, dp + min(self.decompression_pos - dp, 95)):
                        delta = decomp[cur_dp] - prior_val
                        if delta > 2 or delta < -1:
                            break
                        prior_val = decomp[cur_dp]
                        if delta == -1:
                            delta = 3
                        dbuffer[cur_dp - dp] = delta
                    else:
                        cur_dp += 1
                    df_efficiency = (cur_dp - dp) / ((cur_dp - dp) / 4. + 2)
            if re_efficiency >= 4. or re_efficiency > df_efficiency and re_efficiency > 1.:
                comp[cp] = retest + 127
                cp += 1
                comp[cp] = cur_val
                cp += 1
                active_literal_index = -1
                dp += retest
            elif df_efficiency > 1.:
                comp[cp] = cur_dp - dp + 32
                cp += 1
                for delta in range(0, cur_dp - dp, 4):
                    comp[cp] = dbuffer[delta+3] << 6 | dbuffer[delta+2] << 4 | dbuffer[delta+1] << 2 | dbuffer[delta]
                    cp += 1
                active_literal_index = -1
                dp = cur_dp
            else:
                if active_literal_index < 0 or comp[active_literal_index] >= 32:
                    comp[cp] = 0
                    active_literal_index = cp
                    cp += 1
                else:
                    comp[active_literal_index] += 1
                comp[cp] = cur_val
                cp += 1
                dp += 1
        return cp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef long long compress_pbd16(self):
        cdef:
            unsigned char * decomp = <unsigned char *> PyByteArray_AsString(self.decompression_buffer)
            unsigned char * comp = <unsigned char *> PyByteArray_AsString(self.compression_buffer)
            unsigned char retest, carry, i = 0
            char shift
            unsigned char* pb
            unsigned short* pcp2
            unsigned short cur_val, prior_val
            const unsigned short* decomp2 = <unsigned short*>&decomp[0]
            long long decomp_len = self.decompression_pos // 2, active_literal_index = -1, dp2 = 0, cp = 0, cur_dp2
            int delta
            double re_efficiency
            double[3] df_efficiency
            unsigned char[3][256] dbuffer
            long long[3] dc
        assert self.decompression_pos > 0, "The buffer to save is empty."
        while dp2 < decomp_len:
            if cp >= self.compression_pos:
                raise Exception("compression running out of space, try enlarging the compression buffer.")
            retest = 1
            cur_val = decomp2[dp2]
            cur_dp2 = dp2 + 1
            while cur_dp2 < decomp_len and retest < REPEAT_MAX_LEN and decomp2[cur_dp2] == cur_val:
                retest += 1
                cur_dp2 += 1
            re_efficiency = retest / 3.

            if re_efficiency < MAX_EFF[0]:
                df_efficiency[0] = df_efficiency[1] = df_efficiency[2] = 0.
                dc[0] = dc[1] = dc[2] = 0
                if dp2 > 0:
                    for i in range(3):
                        prior_val = decomp2[dp2 - 1]
                        cur_dp2 = dp2
                        for cur_dp2 in range(dp2, dp2 + min(decomp_len - dp2, MAX_LEN[i])):
                            delta = decomp2[cur_dp2] - prior_val
                            if delta > ran[i][1] or delta < ran[i][0]:
                                break
                            prior_val = decomp2[cur_dp2]
                            dbuffer[i][cur_dp2 - dp2] = ran[i][1] - delta if delta < 0 else delta
                        else:
                            cur_dp2 += 1
                        df_efficiency[i] = (cur_dp2 - dp2) / ((cur_dp2 - dp2) * (3 + i) / 16. + 1)
                        dc[i] = cur_dp2
                        if not self.pbd16_full_blood:
                            break
                    else:
                        if df_efficiency[1] > df_efficiency[0]:
                            if df_efficiency[2] > df_efficiency[1]:
                                i = 2
                            else:
                                i = 1
                        elif df_efficiency[2] > df_efficiency[0]:
                            i = 2
            if re_efficiency >= MAX_EFF[0] or re_efficiency > df_efficiency[i] and re_efficiency > 1.:
                comp[cp] = retest + 222
                cp += 1
                pcp2 = <unsigned short*>&comp[cp]
                pcp2[0] = cur_val
                cp += 2
                dp2 += retest
                active_literal_index = -1
            elif df_efficiency[i] > 1.:
                comp[cp] = dc[i] - dp2 + gap[i]
                cp += 1
                carry = 0
                shift = 8
                for cur_dp2 in range(dc[i] - dp2):
                    shift -= shift_bits[i]
                    if shift > 0:
                        carry |= dbuffer[i][cur_dp2] << shift
                    else:
                        carry |= dbuffer[i][cur_dp2] >> -shift
                        comp[cp] = carry
                        cp += 1
                        shift += 8
                        carry = dbuffer[i][cur_dp2] << shift
                else:
                    if shift != 8:
                        comp[cp] = carry
                        cp += 1
                active_literal_index = -1
                dp2 = dc[i]
            else:
                if active_literal_index < 0 or comp[active_literal_index] >= 31:
                    comp[cp] = 0
                    active_literal_index = cp
                    cp += 1
                else:
                    comp[active_literal_index] += 1
                pcp2 = <unsigned short*>&comp[cp]
                pcp2[0] = cur_val
                cp += 2
                dp2 += 1
        return cp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void save(self, path: str | os.PathLike, np.ndarray img):
        """
        :param path: output image path of v3dpbd.
        :param img: 4D numpy array (C,Z,Y,X) of either uint8 or uint16.
        """
        assert img.ndim == 4, "The image has to be 4D"
        assert img.dtype in [np.uint8, np.uint16], "The pixel type has to be uint8 or uint16"
        cdef:
            bytearray header
            int[4] sz = [img.shape[0], img.shape[1], img.shape[2], img.shape[3]]
            int[:] size = sz
            const unsigned char[:] p = str(path).encode('utf-8')
            long long compression_size, channel_len
            short datatype
            FILE * f = fopen(<const char *> &p[0], <char *> 'wb')
        if f is NULL:
            raise Exception("Fail to open file for writing.")
        try:
            endian = '<' if self.endian_sys == LITTLE else '>'
            if img.dtype == np.uint8:
                datatype = 1
            elif img.dtype == np.uint16:
                datatype = 2
            else:
                raise Exception("Unsupported datatype.")
            header = bytearray(FORMAT_KEY + self.endian_sys + struct.pack(f'{endian}hiiii', datatype, *size[::-1]))
            assert fwrite(PyByteArray_AsString(header), HEADER_SIZE, 1, f) == 1, "Header writing failed."
            channel_len = sz[0] * sz[1] * sz[2]
            self.compression_pos = channel_len * sz[3] * datatype * COMPRESSION_ENLARGEMENT
            self.compression_buffer = bytearray(self.compression_pos)
            if datatype == 2 and img.dtype.byteorder not in ['=', endian]:
                img = img.byteswap()
            self.decompression_buffer = bytearray(img.tobytes())
            self.decompression_pos = len(self.decompression_buffer)
            if datatype == 1:
                compression_size = self.compress_pbd8()
            elif datatype == 2:
                compression_size = self.compress_pbd16()
            else:
                raise Exception("Invalid datatype.")
            assert fwrite(<void*>PyByteArray_AsString(self.compression_buffer), compression_size, 1, f) == 1, \
                "Buffer saving failed."
        finally:
            fclose(f)
