"""Test Solid Geometry Utils."""

import numpy as np
import numpy.testing as npt
import pytest

from swcgeom.utils.solid_geometry import (
    find_sphere_line_intersection,
    find_unit_vector_on_plane,
    project_point_on_line,
    project_vector_on_plane,
    project_vector_on_vector,
)


class TestSphereLineIntersection:
    @pytest.mark.parametrize(
        "sphere_center, sphere_radius, line_point_a, line_point_b, expected_intersections",
        [
            # No Intersection Case
            (np.array([0, 0, 0]), 5, np.array([0, -10, 10]), np.array([0, 10, 10]), []),
            # Tangent Case
            (
                np.array([0, 0, 0]),
                5,
                np.array([0, -10, 5]),
                np.array([0, 10, 5]),
                [(0.5, np.array([0, 0, 5]))],
            ),
            # Intersecting Case
            (
                np.array([0, 0, 0]),
                5,
                np.array([-10, 0, 0]),
                np.array([10, 0, 0]),
                [(0.25, np.array([-5, 0, 0])), (0.75, np.array([5, 0, 0]))],
            ),
        ],
    )
    def test_find_sphere_line_intersection(
        self,
        sphere_center,
        sphere_radius,
        line_point_a,
        line_point_b,
        expected_intersections,
    ):
        intersections = find_sphere_line_intersection(
            sphere_center, sphere_radius, line_point_a, line_point_b
        )
        intersections.sort(key=lambda x: x[0])
        assert len(intersections) == len(expected_intersections)
        for actual, expected in zip(intersections, expected_intersections):
            t_actual, point_actual = actual
            t_expected, point_expected = expected
            npt.assert_almost_equal(t_actual, t_expected)
            npt.assert_array_almost_equal(point_actual, point_expected)


class TestFindUnitVectorOnPlane:
    @pytest.mark.parametrize(
        "normal_vec",
        [
            np.array([1, 0, 0]),
            np.array([2, 0, 0]),
            np.array([3, 0, 0]),
        ],
    )
    def test_unit_vector_length(self, normal_vec):
        unit_vec = find_unit_vector_on_plane(normal_vec)
        npt.assert_almost_equal(np.linalg.norm(unit_vec), 1)

    @pytest.mark.parametrize(
        "normal_vec",
        [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ],
    )
    def test_orthogonality_to_normal_vector(self, normal_vec):
        unit_vec = find_unit_vector_on_plane(normal_vec)
        dot_product = np.dot(unit_vec, normal_vec)
        npt.assert_almost_equal(dot_product, 0)

    @pytest.mark.parametrize(
        "normal_vec",
        [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ],
    )
    def test_handles_different_inputs(self, normal_vec):
        unit_vec = find_unit_vector_on_plane(normal_vec)
        npt.assert_almost_equal(np.linalg.norm(unit_vec), 1)
        npt.assert_almost_equal(np.dot(unit_vec, normal_vec), 0)


class TestProjectVectorOnLine:
    @pytest.mark.parametrize(
        "point_a, direction_vector, point_p, expected",
        [
            ([0, 0], [1, 0], [3, 4], [3, 0]),  # horizontal line
            ([0, 0], [0, 1], [3, 4], [0, 4]),  # vertical line
            ([1, 2], [2, 3], [1, 2], [1, 2]),  # point on line
            ([1, 1], [1, 1], [2, 3], [2.5, 2.5]),  # ramdom point
        ],
    )
    def test_project(self, point_a, direction_vector, point_p, expected):
        projection = project_point_on_line(point_a, direction_vector, point_p)
        npt.assert_almost_equal(projection, expected)


class TestProjectPointOnVec:
    @pytest.mark.parametrize(
        "vec, target, expected",
        [
            ([3, 4, 5], [1, 0, 0], [3, 0, 0]),  # porject to x-axis
            ([3, 4, 5], [0, 1, 0], [0, 4, 0]),  # porject to y-axis
            ([3, 4, 5], [0, 0, 1], [0, 0, 5]),  # porject to z-axis
            ([3, 4, 5], [1, 1, 1], [4, 4, 4]),  # porject to a vector
            ([3, 4, 5], [3, 4, 5], [3, 4, 5]),  # vector on the plene
            ([1, 0, 0], [0, 1, 0], [0, 0, 0]),  # vector is orthogonal to the plane
            ([0, 0, 0], [1, 1, 1], [0, 0, 0]),  # vector is zero
        ],
    )
    def test_project(self, vec, target, expected):
        projection = project_vector_on_vector(vec, target)
        npt.assert_almost_equal(projection, expected)


class TestProjectPointOnPlane:
    @pytest.mark.parametrize(
        "vec, plane_normal, expected",
        [
            ([3, 4, 5], [0, 0, 1], [3, 4, 0]),  # porject to xy-plane
            ([3, 4, 5], [1, 0, 0], [0, 4, 5]),  # porject to yz-plane
            ([3, 4, 5], [0, 1, 0], [3, 0, 5]),  # porject to xz-plane
            ([3, 4, 5], [1, 1, 1], [-1, 0, 1]),  # porject to a plane
            ([3, 4, 0], [0, 0, 1], [3, 4, 0]),  # vector on the plane
            ([1, 0, 0], [1, 0, 0], [0, 0, 0]),  # vector is orthogonal to the plane
            ([0, 0, 0], [1, 1, 1], [0, 0, 0]),  # vector is zero
        ],
    )
    def test_project(self, vec, plane_normal, expected):
        projection = project_vector_on_plane(vec, plane_normal)
        npt.assert_almost_equal(projection, expected)
