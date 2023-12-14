"""Test Geometry Object."""

import numpy as np

from swcgeom.utils.geometry_object import GeomFrustumCone, GeomSphere


def test_sphere_volume():
    # (sphere radius, except_volume)
    test_cases = [
        # fmt:off
        (9,   4 / 3 * 9**3 * np.pi),
        (5,   4 / 3 * 5**3 * np.pi),
        (3,   4 / 3 * 3**3 * np.pi),
        (0, 0.0                   ),
        # fmt:on
    ]

    for radius, except_volume in test_cases:
        volume = GeomSphere.calc_volume(radius)
        assert np.isclose(
            volume, except_volume
        ), f"Test Failed: Expected volume: {except_volume}, Got: {volume}"


def test_spherical_cap_volume():
    # (sphere radius, cap height, except_volume)
    test_cases = [
        # fmt:off
        (9, 5, 550.0 / 3.0 * np.pi),
        (5, 2,  52.0 / 3.0 * np.pi),
        (3, 1,   8.0 / 3.0 * np.pi),
        (3, 3,  18.0       * np.pi),
        (3, 4,  80.0 / 3.0 * np.pi),
        (4, 8, 256.0 / 3.0 * np.pi),
        (4, 0,   0.0              ),
        # fmt:on
    ]

    for radius, height, except_volume in test_cases:
        volume = GeomSphere.calc_volume_spherical_cap(radius, height)
        assert np.isclose(
            volume, except_volume
        ), f"Test Failed: Expected volume: {except_volume}, Got: {volume}"

    # Test the sum of two spherical caps is equal to the volume of the sphere
    for r in range(0, 10):
        for h in range(0, r + 1):
            v1 = GeomSphere.calc_volume_spherical_cap(r, h)
            v2 = GeomSphere.calc_volume_spherical_cap(r, 2 * r - h)
            v = v1 + v2
            v_shpere = GeomSphere.calc_volume(r)
            assert np.isclose(
                v, v_shpere
            ), f"Test Failed: Expected volume: {v_shpere}, Got: {v}"


def test_intersect_volume_sphere_sphere():
    # Create spheres
    sphere1 = GeomSphere((0.0, 0.0, 0.0), 2)  # Center at origin, radius 2
    sphere2 = GeomSphere((3.0, 0.0, 0.0), 2)  # Center at (3,0,0), radius 2
    sphere3 = GeomSphere((0.0, 0.0, 0.0), 1)  # Center at origin, radius 1
    sphere4 = GeomSphere((0.0, 5.0, 0.0), 2)  # Center at (0,5,0), radius 2

    # Overlapping spheres - expected volume should be calculated
    volume1 = sphere1.get_intersect_volume(sphere2)
    expected_volume1 = GeomSphere.calc_volume_spherical_cap(2, 0.5) * 2
    assert np.isclose(
        volume1, expected_volume1
    ), "Test Failed: Overlapping spheres should have a non-zero intersecting volume"

    # One sphere completely inside the other - expected volume should be the volume of the smaller sphere
    volume2 = sphere1.get_intersect_volume(sphere3)
    expected_volume2 = GeomSphere.calc_volume(1)
    assert np.isclose(
        volume2, expected_volume2
    ), f"Test Failed: The intersecting volume should be equal to the volume of the smaller sphere. Expected: {expected_volume2}, Got: {volume2}"

    # Non-intersecting spheres - expected volume should be zero
    volume3 = sphere1.get_intersect_volume(sphere4)
    assert (
        volume3 == 0
    ), "Test Failed: Non-intersecting spheres should have zero intersecting volume"


def test_intersect_volume_sphere_frustum_cone():
    # The height of the frustum is bigger than the sum of two sphere radius
    frustum1 = GeomFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 8.0), 2)
    sphere1a = GeomSphere((0.0, 0.0, 0.0), 4)
    sphere1b = GeomSphere((0.0, 0.0, 8.0), 2)

    volume1a = sphere1a.get_intersect_volume(frustum1)  # TODO: expected volume?
    assert volume1a > 0, "Test Failed: Expected volume should be non-zero"

    volume1b = sphere1b.get_intersect_volume(frustum1)
    excepted_volume1b = sphere1b.get_volume() / 2
    assert np.isclose(
        volume1b, excepted_volume1b
    ), "Test Failed: Expected volume should be half of the sphere volume"
    # TODO

    # The height of the frustum is smaller than the sum of two sphere radius
    frustum2 = GeomFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 5.0), 2)
    sphere2a = GeomSphere((0.0, 0.0, 0.0), 4)
    sphere2b = GeomSphere((0.0, 0.0, 5.0), 2)

    volume2a = sphere2a.get_intersect_volume(frustum2)  # TODO: expected volume?
    assert volume2a > 0, "Test Failed: Expected volume should be non-zero"

    volume2b = sphere2b.get_intersect_volume(frustum2)
    excepted_volume2b = sphere2b.get_volume() / 2
    assert np.isclose(
        volume2b, excepted_volume2b
    ), "Test Failed: Expected volume should be half of the sphere volume"

    # The height of the frustum is smaller than the minimal of two sphere radius
    frustum3 = GeomFrustumCone((0.0, 0.0, 0.0), 4, (0.0, 0.0, 1.0), 2)
    sphere3a = GeomSphere((0.0, 0.0, 0.0), 4)
    sphere3b = GeomSphere((0.0, 0.0, 1.0), 2)

    volume3a = sphere3a.get_intersect_volume(frustum3)  # TODO: expected volume?
    assert volume3a > 0, "Test Failed: Expected volume should be non-zero"

    volume3b = sphere3b.get_intersect_volume(frustum3)
    excepted_volume3b = sphere3b.get_volume_spherical_cap(
        2
    ) - sphere3b.get_volume_spherical_cap(1)
    assert np.isclose(
        volume3b, excepted_volume3b
    ), "Test Failed: Expected volume should be half of the sphere volume"
