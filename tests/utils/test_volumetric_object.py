"""Test Geometry Object."""

import numpy as np
import numpy.testing as npt
import pytest

from swcgeom.utils.volumetric_object import VolFrustumCone, VolSphere


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
        volume = VolSphere.calc_volume(sphere_radius)
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
        volume = VolSphere.calc_volume_spherical_cap(sphere_radius, cap_height)
        npt.assert_allclose(volume, excepted_volume)

    @pytest.mark.parametrize(
        "r, h",
        [(4, 0), (4, 3), (4, 4), (7, 0), (7, 2), (7, 7)],
    )
    def test_spherical_cap_volume_complementation(self, r, h):
        # Test the sum of two spherical caps is equal to the volume of the sphere
        v1 = VolSphere.calc_volume_spherical_cap(r, h)
        v2 = VolSphere.calc_volume_spherical_cap(r, 2 * r - h)
        v = v1 + v2
        v_shpere = VolSphere.calc_volume(r)
        npt.assert_allclose(v, v_shpere)


class TestSphere2Intersection:
    def test_overlapping(self):
        sphere1 = VolSphere((0.0, 0.0, 0.0), 2)  # Center at origin, radius 2
        sphere2 = VolSphere((3.0, 0.0, 0.0), 2)  # Center at (3,0,0), radius 2
        volume = sphere1.intersect(sphere2).get_volume()
        expected_volume = VolSphere.calc_volume_spherical_cap(2, 0.5) * 2
        npt.assert_allclose(volume, expected_volume)

    def test_contains(self):
        sphere1 = VolSphere((0.0, 0.0, 0.0), 2)  # Center at origin, radius 2
        sphere2 = VolSphere((0.0, 0.0, 0.0), 1)  # Center at origin, radius 1
        volume = sphere1.intersect(sphere2).get_volume()
        expected_volume = VolSphere.calc_volume(1)
        npt.assert_allclose(volume, expected_volume)

    def test_no_intersection(self):
        sphere1 = VolSphere((0.0, 0.0, 0.0), 2)  # Center at origin, radius 2
        sphere2 = VolSphere((0.0, 5.0, 0.0), 2)  # Center at (0,5,0), radius 2
        volume = sphere1.intersect(sphere2).get_volume()
        assert volume == 0


class TestSphereFrustumConeIntersection:
    def test_high_frustum(self):
        # The height of the frustum is bigger than the sum of two sphere radius
        frustum = VolFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 8.0), 2)
        sphere1 = VolSphere((0.0, 0.0, 0.0), 4)
        sphere2 = VolSphere((0.0, 0.0, 8.0), 2)

        volume1 = sphere1.intersect(frustum).get_volume()
        assert volume1 > 0  # TODO: expected volume?

        volume2 = sphere2.intersect(frustum).get_volume()
        excepted_volume2 = sphere2.get_volume() / 2
        npt.assert_allclose(volume2, excepted_volume2)
        # TODO

    def test_low_frustum(self):
        # The height of the frustum is smaller than the sum of two sphere radius
        frustum = VolFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 5.0), 2)
        sphere1 = VolSphere((0.0, 0.0, 0.0), 4)
        sphere2 = VolSphere((0.0, 0.0, 5.0), 2)

        volume1 = sphere1.intersect(frustum).get_volume()
        assert volume1 > 0  # TODO: expected volume?

        volume2 = sphere2.intersect(frustum).get_volume()
        excepted_volume2 = sphere2.get_volume() / 2
        npt.assert_allclose(volume2, excepted_volume2)

    def test_much_low_frustum(self):
        # The height of the frustum is smaller than the minimal of two sphere radius
        frustum = VolFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 1.0), 2)
        sphere1 = VolSphere((0.0, 0.0, 0.0), 4)
        sphere2 = VolSphere((0.0, 0.0, 1.0), 2)

        volume3a = sphere1.intersect(frustum).get_volume()
        assert volume3a > 0  # TODO: expected volume?

        volume3b = sphere2.intersect(frustum).get_volume()
        excepted_volume3b = sphere2.get_volume_spherical_cap(
            2
        ) - sphere2.get_volume_spherical_cap(1)
        npt.assert_allclose(volume3b, excepted_volume3b)

    def test_almost_cylinder(self):
        # The edge case where the frustum is like a cylinder
        sphere = VolSphere((391.58, 324.97, -12.89), 0.493507)
        frustum_cone = VolFrustumCone(
            (391.58, 324.97, -12.89), 0.493507, (388.07, 320.41, -13.57), 0.493506
        )
        assert sphere.intersect(frustum_cone).get_volume() > 0


class TestFrustumCone2DiffSphere:
    def test_contains(self):
        sphere = VolSphere((0, 0, 0), 1)
        frustum1 = VolFrustumCone((0, 0, 0), 1, (0, 0, 8), 1)
        frustum2 = VolFrustumCone((0, 0, 0), 1, (0, 0, 4), 1)
        volume = frustum1.intersect(frustum2).subtract(sphere).get_volume()
        excepted_volume = (4 - 2 / 3) * np.pi
        npt.assert_allclose(volume, excepted_volume, rtol=5e-2)
