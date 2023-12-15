"""Test Geometry Object."""

import numpy as np
import numpy.testing as npt
import pytest

from swcgeom.utils.geometry_object import GeomFrustumCone, GeomSphere


class TestSphere:
    @pytest.mark.parametrize(
        "sphere_radius, except_volume",
        [
            # fmt:off
            (9,   4 / 3 * 9**3 * np.pi),
            (5,   4 / 3 * 5**3 * np.pi),
            (3,   4 / 3 * 3**3 * np.pi),
            (0, 0.0                   ),
            # fmt:on
        ],
    )
    def test_sphere_volume(self, sphere_radius, except_volume):
        volume = GeomSphere.calc_volume(sphere_radius)
        npt.assert_allclose(volume, except_volume)

    @pytest.mark.parametrize(
        "sphere_radius, cap_height, excepted_volume",
        [
            # fmt:off
            (9, 5, 550.0 / 3.0 * np.pi),
            (5, 2,  52.0 / 3.0 * np.pi),
            (3, 1,   8.0 / 3.0 * np.pi),
            (3, 3,  18.0       * np.pi),
            (3, 4,  80.0 / 3.0 * np.pi),
            (4, 8, 256.0 / 3.0 * np.pi),
            (4, 0,   0.0              ),
            # fmt:on
        ],
    )
    def test_spherical_cap_volume(self, sphere_radius, cap_height, excepted_volume):
        volume = GeomSphere.calc_volume_spherical_cap(sphere_radius, cap_height)
        npt.assert_allclose(volume, excepted_volume)

    @pytest.mark.parametrize(
        "r, h",
        [(4, 0), (4, 3), (4, 4), (7, 0), (7, 2), (7, 7)],
    )
    def test_spherical_cap_volume_complementation(self, r, h):
        # Test the sum of two spherical caps is equal to the volume of the sphere
        v1 = GeomSphere.calc_volume_spherical_cap(r, h)
        v2 = GeomSphere.calc_volume_spherical_cap(r, 2 * r - h)
        v = v1 + v2
        v_shpere = GeomSphere.calc_volume(r)
        npt.assert_allclose(v, v_shpere)


class TestIntersectVolumeSphereSphere:
    def test_overlapping(self):
        sphere1 = GeomSphere((0.0, 0.0, 0.0), 2)  # Center at origin, radius 2
        sphere2 = GeomSphere((3.0, 0.0, 0.0), 2)  # Center at (3,0,0), radius 2
        volume1 = sphere1.get_intersect_volume(sphere2)
        expected_volume1 = GeomSphere.calc_volume_spherical_cap(2, 0.5) * 2
        npt.assert_allclose(volume1, expected_volume1)

    def test_contains(self):
        sphere1 = GeomSphere((0.0, 0.0, 0.0), 2)  # Center at origin, radius 2
        sphere2 = GeomSphere((0.0, 0.0, 0.0), 1)  # Center at origin, radius 1
        volume = sphere1.get_intersect_volume(sphere2)
        expected_volume = GeomSphere.calc_volume(1)
        npt.assert_allclose(volume, expected_volume)

    def test_no_intersection(self):
        sphere1 = GeomSphere((0.0, 0.0, 0.0), 2)  # Center at origin, radius 2
        sphere2 = GeomSphere((0.0, 5.0, 0.0), 2)  # Center at (0,5,0), radius 2
        volume = sphere1.get_intersect_volume(sphere2)
        assert volume == 0


class TestIntersectVolumeSphereFrustumCone:
    def test_high_frustum(self):
        # The height of the frustum is bigger than the sum of two sphere radius
        frustum = GeomFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 8.0), 2)
        sphere1 = GeomSphere((0.0, 0.0, 0.0), 4)
        sphere2 = GeomSphere((0.0, 0.0, 8.0), 2)

        volume1 = sphere1.get_intersect_volume(frustum)
        assert volume1 > 0  # TODO: expected volume?

        volume2 = sphere2.get_intersect_volume(frustum)
        excepted_volume2 = sphere2.get_volume() / 2
        npt.assert_allclose(volume2, excepted_volume2)
        # TODO

    def test_low_frustum(self):
        # The height of the frustum is smaller than the sum of two sphere radius
        frustum = GeomFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 5.0), 2)
        sphere1 = GeomSphere((0.0, 0.0, 0.0), 4)
        sphere2 = GeomSphere((0.0, 0.0, 5.0), 2)

        volume1 = sphere1.get_intersect_volume(frustum)
        assert volume1 > 0  # TODO: expected volume?

        volume2 = sphere2.get_intersect_volume(frustum)
        excepted_volume2 = sphere2.get_volume() / 2
        npt.assert_allclose(volume2, excepted_volume2)

    def test_much_low_frustum(self):
        # The height of the frustum is smaller than the minimal of two sphere radius
        frustum = GeomFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 1.0), 2)
        sphere1 = GeomSphere((0.0, 0.0, 0.0), 4)
        sphere2 = GeomSphere((0.0, 0.0, 1.0), 2)

        volume3a = sphere1.get_intersect_volume(frustum)
        assert volume3a > 0  # TODO: expected volume?

        volume3b = sphere2.get_intersect_volume(frustum)
        excepted_volume3b = sphere2.get_volume_spherical_cap(
            2
        ) - sphere2.get_volume_spherical_cap(1)
        npt.assert_allclose(volume3b, excepted_volume3b)

    def test_almost_cylinder(self):
        # The edge case where the frustum is like a cylinder
        sphere = GeomSphere((391.58, 324.97, -12.89), 0.493507)
        frustum_cone = GeomFrustumCone(
            (391.58, 324.97, -12.89), 0.493507, (388.07, 320.41, -13.57), 0.493506
        )
        assert sphere.get_intersect_volume(frustum_cone) > 0
