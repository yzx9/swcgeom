import numpy as np

from swcgeom.utils.numpy_helper import padding1d


class TestPadding1d:
    def test_no_input_vector(self):
        n = 5
        result = padding1d(n, None)
        expected = np.zeros(n)
        np.testing.assert_array_equal(result, expected)

    def test_input_length_equals_n(self):
        n = 5
        v = np.arange(n)
        result = padding1d(n, v)
        np.testing.assert_array_equal(result, v)

    def test_input_length_less_than_n(self):
        n = 5
        v = np.arange(3)
        result = padding1d(n, v)
        expected = np.array([0, 1, 2, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_input_length_greater_than_n(self):
        n = 3
        v = np.arange(5)
        result = padding1d(n, v)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_specify_dtype(self):
        n = 5
        v = np.arange(3)
        result = padding1d(n, v, dtype=np.int32)
        assert np.issubdtype(result.dtype, np.int32)

    def test_input_is_multidimensional(self):
        n = 5
        v = np.array([[1, 2, 3], [4, 5, 6]])
        try:
            padding1d(n, v)
            assert False
        except:
            pass
