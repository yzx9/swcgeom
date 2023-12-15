"""Analysis of volume of a SWC tree."""

from typing import List

import numpy as np

from swcgeom.core import Tree
from swcgeom.utils import GeomFrustumCone, GeomSphere

__all__ = ["get_volume"]


def get_volume(tree: Tree):
    """Get the volume of the tree.

    Parameters
    ----------
    tree : Tree
        SWC tree.

    Returns
    -------
    volume : float
        Volume of the tree.

    Notes
    -----
    The SWC format is a method for representing neurons, which includes
    both the radius of individual points and their interconnectivity.
    Consequently, there are multiple distinct approaches to
    representation within this framework.

    Currently, we support a standard approach to volume calculation.
    This method involves treating each node as a sphere and
    representing the connections between them as truncated cone-like
    structures, or frustums, with varying radii at their top and bottom
    surfaces.

    We welcome additional representation methods through pull requests.
    """
    volume = 0.0

    def leave(n: Tree.Node, children: List[GeomSphere]) -> GeomSphere:
        sphere = GeomSphere(n.xyz(), n.r)
        cones = [GeomFrustumCone(n.xyz(), n.r, c.center, c.radius) for c in children]

        v = sphere.get_volume()
        v += sum(fc.get_volume() for fc in cones)
        v -= sum(sphere.get_intersect_volume(fc) for fc in cones)
        v -= sum(s.get_intersect_volume(fc) for s, fc in zip(children, cones))
        v += sum(s.get_intersect_volume(sphere) for s in children)

        # TODO
        # remove volume of intersection between frustum cones
        # v -= sum(
        #     fc1.get_intersect_volume(fc2)
        #     for fc1 in frustum_cones
        #     for fc2 in frustum_cones
        #     if fc1 != fc2
        # )

        nonlocal volume
        volume += v
        return sphere

    tree.traverse(leave=leave)
    return volume
