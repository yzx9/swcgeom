"""Analysis of volume of a SWC tree."""

from typing import List

from swcgeom.core import Tree
from swcgeom.utils import VolFrustumCone, VolSphere

__all__ = ["get_volume"]


def get_volume(tree: Tree) -> float:
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

    def leave(n: Tree.Node, children: List[VolSphere]) -> VolSphere:
        sphere = VolSphere(n.xyz(), n.r)
        cones = [VolFrustumCone(n.xyz(), n.r, c.center, c.radius) for c in children]

        v = sphere.get_volume()
        v += sum(fc.get_volume() for fc in cones)
        v -= sum(sphere.intersect(fc).get_volume() for fc in cones)
        v -= sum(s.intersect(fc).get_volume() for s, fc in zip(children, cones))
        v += sum(s.intersect(sphere).get_volume() for s in children)
        v -= sum(
            cones[i].intersect(cones[j]).subtract(sphere).get_volume()
            for i in range(len(cones))
            for j in range(i + 1, len(cones))
        )

        nonlocal volume
        volume += v
        return sphere

    tree.traverse(leave=leave)
    return volume
