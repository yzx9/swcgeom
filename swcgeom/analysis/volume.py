"""Analysis of volume of a SWC tree."""

from typing import Dict, List, Literal, Tuple

from swcgeom.core import Tree
from swcgeom.utils import VolFrustumCone, VolObject, VolSphere

__all__ = ["get_volume"]

ACCURACY_LEVEL = Literal["low", "middle", "high"]
ACCURACY_LEVELS: Dict[ACCURACY_LEVEL, int] = {"low": 3, "middle": 5, "high": 8}


def get_volume(
    tree: Tree,
    *,
    method: Literal["frustum_cone"] = "frustum_cone",
    accuracy: int | ACCURACY_LEVEL = "middle",
) -> float:
    """Get the volume of the tree.

    Parameters
    ----------
    tree : Tree
        Neuronal tree.
    method : {"frustum_cone"}, optional
        Method for volume calculation.
    accuracy : int or {"low", "middle", "high"}, optional
        Accuracy level for volume calculation. The higher the accuracy,
        the more accurate the volume calculation, but the slower the
        calculation. The accuracy level can be specified either as an
        integer or as a string.

        The string values correspond to the following accuracy levels:

        - "low": 3
        - "middle": 5
        - "high": 8

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

    if isinstance(accuracy, str):
        accuracy = ACCURACY_LEVELS[accuracy]

    assert 0 < accuracy <= 10

    match method:
        case "frustum_cone":
            return _get_volume_frustum_cone(tree, accuracy=accuracy)
        case _:
            raise ValueError(f"Unsupported method: {method}")


def _get_volume_frustum_cone(tree: Tree, *, accuracy: int) -> float:
    """Get the volume of the tree using the frustum cone method.

    Parameters
    ----------
    tree : Tree
        Neuronal tree.
    accuracy : int
        1 : Sphere only
        2 : Sphere and Frustum Cone
        3 : Sphere, Frustum Cone, and intersection in single-branch
        5 : Above and Sphere-Frustum Cone intersection in multi-branch
        10 : Fully calculated by Monte Carlo method
    """

    if accuracy == 10:
        return _get_volume_frustum_cone_mc_only(tree)

    volume = 0.0

    def leave(n: Tree.Node, children: List[VolSphere]) -> VolSphere:
        sphere = VolSphere(n.xyz(), n.r)
        cones = [VolFrustumCone(n.xyz(), n.r, c.center, c.radius) for c in children]

        v = sphere.get_volume()
        if accuracy >= 2:
            v += sum(fc.get_volume() for fc in cones)

        if accuracy >= 3:
            v -= sum(sphere.intersect(fc).get_volume() for fc in cones)
            v -= sum(s.intersect(fc).get_volume() for s, fc in zip(children, cones))
            v += sum(s.intersect(sphere).get_volume() for s in children)

        if accuracy >= 5:
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


_R = Tuple[VolObject, VolSphere]


def _get_volume_frustum_cone_mc_only(tree: Tree) -> float:
    def leave(n: Tree.Node, children: List[_R]) -> _R:
        sphere = VolSphere(n.xyz(), n.r)
        obj = sphere
        for o, c in children:
            obj = obj.union(o)
            obj = obj.union(VolFrustumCone(n.xyz(), n.r, c.center, c.radius))

        return obj, sphere

    obj, _ = tree.traverse(leave=leave)
    # TODO: estimate the number of samples needed
    volume = obj.get_volume(n_samples=100_000_000)
    return volume
