"""L-Measure analysis."""

import math
from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt

from swcgeom.core import Branch, Compartment, Node, Tree
from swcgeom.utils import angle

__all__ = ["LMeasure"]


class LMeasure:
    """L-Measure analysis.

    The L-Measure analysis provide a set of morphometric features for
    multiple levels analysis, as described in the paper [1]_

    References
    ----------
    .. [1] Scorcioni R, Polavaram S, Ascoli GA. L-Measure: a
       web-accessible tool for the analysis, comparison and search of
       digital reconstructions of neuronal morphologies. Nat Protoc.
       2008;3(5):866-76. doi: 10.1038/nprot.2008.51. PMID: 18451794;
       PMCID: PMC4340709.

    See Also
    --------
    L-Measure: http://cng.gmu.edu:8080/Lm/help/index.htm
    """

    def __init__(self, compartment_point: Literal[0, -1] = -1):
        """
        Parameters
        ----------
        compartment_point : 0 | -1, default=-1
            The point of the compartment to be used for the measurements.
            If 0, the first point of the compartment is used. If -1, the
            last point of the compartment is used.
        """
        super().__init__()
        self.compartment_point = compartment_point

    # Topological measurements

    def n_stems(self, tree: Tree) -> int:
        """Number of stems that is connected to soma.

        This function returns the number of stems attached to the soma.
        When the type of the Compartment changes from type=1 to others
        it is labeled a stem. These stems can also be considered as
        independent subtrees for subtree level analysis.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/N_stems.htm
        """
        return len(tree.soma().children())

    def n_bifs(self, tree: Tree) -> int:
        """Number of bifurcations.

        This function returns the number of bifurcations for the given
        input neuron. A bifurcation point has two daughters.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/N_bifs.htm
        """
        return len(tree.get_bifurcations())

    def n_branch(self, tree: Tree) -> int:
        """Number of branches.

        This function returns the number of branches in the given input
        neuron. A branch is one or more compartments that lie between
        two branching points or between one branching point and a
        termination point.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/N_branch.htm
        """
        return len(tree.get_branches())

    def n_tips(self, tree: Tree) -> int:
        """Number of terminal tips.

        This function returns the number of terminal tips for the given
        input neuron. This function counts the number of compartments
        that terminate as terminal endpoints.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/N_tips.htm
        """
        return len(tree.get_tips())

    def terminal_segment(self) -> int:
        """Terminal Segment.

        TerminalSegment is the branch that ends as a terminal branch.
        This function returns "1" for all the compartments in the
        terminal branch.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/TerminalSegment.htm
        """
        raise NotImplementedError()

    def branch_pathlength(self, branch: Branch) -> float:
        """Length of the branch.

        This function returns the sum of the length of all compartments
        forming the giveN_branch.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Branch_pathlength.htm
        """
        # ? as a topological measurement, this should accept a tree
        return branch.length()

    def contraction(self, branch: Branch) -> float:
        """Contraction of the branch.

        This function returns the ratio between Euclidean distance of a
        branch and its path length.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Contraction.htm
        """
        # ? as a topological measurement, this should accept a tree
        euclidean = branch[0].distance(branch[-1])
        return euclidean / branch.length()

    def fragmentation(self, branch: Branch) -> int:
        """Number of compartments.

        This function returns the total number of compartments that
        constitute a branch between two bifurcation points or between a
        bifurcation point and a terminal tip.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Fragmentation.htm
        """
        # ? as a topological measurement, this should accept a tree
        return branch.number_of_edges()

    def partition_asymmetry(self, n: Tree.Node) -> float:
        """Partition asymmetry.

        This is computed only on bifurcation. If n1 is the number of
        tips on the left and n2 on the right. Asymmetry return
        abs(n1-n2)/(n1+n2-2).

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Partition_asymmetry.htm
        """
        children = n.children()
        assert (
            len(children) == 2
        ), "Partition asymmetry is only defined for bifurcations"
        n1 = len(children[0].subtree().get_tips())
        n2 = len(children[1].subtree().get_tips())
        return abs(n1 - n2) / (n1 + n2 - 2)

    def fractal_dim(self):
        """Fractal dimension.

        Fractal dimension (D) of neuronal branches is computedas the
        slope of linear fit of regression line obtained from the
        log-log plot of Path distance vs Euclidean distance.

        This method of measuring the fractal follows the reference
        given below by Marks & Burke, J Comp Neurol. 2007. [1]_
        - When D = 1, the particle moves in a straight line.
        - When D = 2, the motion is a space-filling random walk
        - When D is only slightly larger than 1, the particle
          trajectory resembles a country road or a dendrite branch.

        References
        ----------
        .. [1] Marks WB, Burke RE. Simulation of motoneuron morphology
           in three dimensions. I. Building individual dendritic trees.
           J Comp Neurol. 2007 Aug 10;503(5):685-700.
           doi: 10.1002/cne.21418. PMID: 17559104.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Fractal_Dim.htm
        """
        raise NotImplementedError()

    # Branch level measurements (thickness and taper)

    def taper_1(self, branch: Branch) -> float:
        """The Burke Taper.

        This function returns the Burke Taper. This function is
        measured between two bifurcation points.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Taper_1.htm
        """
        da, db = 2 * branch[0].r, 2 * branch[-1].r
        return (da - db) / branch.length()

    def taper_2(self, branch: Branch) -> float:
        """The Hillman taper.

        This function returns the Hillman taper. This is measured
        between two bifurcation points.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Taper_2.htm
        """
        da, db = 2 * branch[0].r, 2 * branch[-1].r
        return (da - db) / da

    def daughter_ratio(self):
        """Daughter ratio.

        The function returns the ratio between the bigger daughter and
        the other one. A daughter is the next immediate compartment
        connected to a bifurcation point.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Daughter_Ratio.htm
        """
        raise NotImplementedError()

    def parent_daughter_ratio(self) -> float:
        """Parent daughter ratio.

        This function returns the ratio between the diameter of a
        daughter and its father. One values for each daughter is
        returned at each bifurcation point.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Parent_Daughter_Ratio.htm
        """
        raise NotImplementedError()

    def rall_power(self, bif: Tree.Node) -> float:
        """Rall Power.

        Rall value is computed as the best value that fits the equation
        (Bif_Dia)^Rall=(Daughter1_dia^Rall+Daughter2_dia^Rall).
        According to Rall’s rule we compute rall’s power by linking the
        diameter of two daughter branches to the diameter of the
        bifurcating parent. We compute the best fit rall’s power within
        the boundary values of [0, 5] at incremental steps of 1000
        compartments. The final rall value is the idealistic n value
        that can propagate the signal transmission without loss from
        the starting point to the terminal point in a cable model
        assumption.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Rall_Power.htm
        """

        rall_power, _, _, _ = self._rall_power(bif)
        return rall_power

    def _rall_power_d(self, bif: Tree.Node) -> Tuple[float, float, float]:
        children = bif.children()
        assert len(children) == 2, "Rall Power is only defined for bifurcations"
        parent = bif.parent()
        assert parent is not None, "Rall Power is not defined for root"

        dp = 2 * parent.r
        da, db = 2 * children[0].r, 2 * children[1].r
        return dp, da, db

    def _rall_power(self, bif: Tree.Node) -> Tuple[float, float, float, float]:
        dp, da, db = self._rall_power_d(bif)
        start, stop, step = 0, 5, 5 / 1000
        xs = np.arange(start, stop, step)
        ys = (da**xs + db**xs) - dp**xs
        return xs[np.argmin(ys)], dp, da, db

    def pk(self, bif: Tree.Node) -> float:
        """Ratio of rall power increased.

        After computing the average value for Rall_Power, this function
        returns the ratio of (d1^rall+d2^rall)/(bifurcDiam^rall).

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Pk.htm
        """

        rall_power, dp, da, db = self._rall_power(bif)
        return (da**rall_power + db**rall_power) / dp**rall_power

    def pk_classic(self, bif: Tree.Node) -> float:
        """Ratio of rall power increased with fixed rall power 1.5.

        This function returns the same value as Pk, but with Rall_Power
        sets to 1.5.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Pk_classic.htm
        """

        dp, da, db = self._rall_power_d(bif)
        rall_power = 1.5
        return (da**rall_power + db**rall_power) / dp**rall_power

    def pk_2(self, bif: Tree.Node) -> float:
        """Ratio of rall power increased with fixed rall power 2.

        This function returns the same value as Pk, but with Rall_Power
        sets to 2 .

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Pk_2.htm
        """

        dp, da, db = self._rall_power_d(bif)
        rall_power = 2
        return (da**rall_power + db**rall_power) / dp**rall_power

    def bif_ampl_local(self, bif: Tree.Node) -> float:
        """Bifuraction angle.

        Given a bifurcation, this function returns the angle between
        the first two compartments (in degree).

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Bif_ampl_local.htm
        """

        v1, v2 = self._bif_vector_local(bif)
        return np.degrees(angle(v1, v2))

    def bif_ampl_remote(self, bif: Tree.Node) -> float:
        """Bifuraction angle.

        This function returns the angle between two bifurcation points
        or between bifurcation point and terminal point or between two
        terminal points.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Bif_ampl_remote.htm
        """

        v1, v2 = self._bif_vector_local(bif)
        return np.degrees(angle(v1, v2))

    def bif_tilt_local(self, bif: Tree.Node) -> float:
        """Bifuarcation tilt.

        This function returns the angle between the previous
        compartment of bifurcating father and the two daughter
        compartments of the same bifurcation. The smaller of the two
        angles is returned as the result.

        Tilt is measured as outer angle between parent and child
        compartments. L-Measure returns smaller angle of the two
        children, but intuitively this can be viewed as the angle of
        deflection of the parent orientation compared to the
        orientation of the mid line of the bifurcation amplitude angle.

        (i.e.) new tilt(NT) = pi - 1/2 Bif_amp_remote - old tilt(OT)

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Bif_tilt_local.htm
        """

        parent = bif.parent()
        assert parent is not None, "Bifurcation tilt is not defined for root"
        v = parent.xyz() - bif.xyz()
        v1, v2 = self._bif_vector_local(bif)

        angle1 = np.degrees(angle(v, v1))
        angle2 = np.degrees(angle(v, v2))
        return min(angle1, angle2)

    def bif_tilt_remote(self, bif: Tree.Node) -> float:
        """Bifuarcation tilt.

        This function returns the angle between the previous father
        node of the current bifurcating father and its two daughter
        nodes. A node is a terminating point or a bifurcation point or
        a root point. Smaller of the two angles is returned as the
        result. This angle is not computed for the root node.

        Tilt is measured as outer angle between parent and child
        compartments. L-Measure returns smaller angle of the two
        children, but intuitively this can be viewed as the angle of
        deflection of the parent orientation compared to the
        orientation of the mid line of the bifurcation amplitude angle.

        (i.e.) new tilt(NT) = pi - 1/2 Bif_amp_remote - old tilt(OT)

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Bif_tilt_remote.htm
        """

        parent = bif.parent()
        assert parent is not None, "Bifurcation tilt is not defined for root"
        v = parent.xyz() - bif.xyz()
        v1, v2 = self._bif_vector_remote(bif)

        angle1 = np.degrees(angle(v, v1))
        angle2 = np.degrees(angle(v, v2))
        return min(angle1, angle2)

    def bif_torque_local(self, bif: Tree.Node) -> float:
        """Bifurcation torque.

        This function returns the angle between the plane of previous
        bifurcation and the current bifurcation. Bifurcation plane is
        identified by the two daughter compartments leaving the
        bifurcation.

        A torque is the inner angle measured between two planes
        (current & parent) of bifurcations. Plane DCE has current
        bifurcation and plane CAB has parent bifurcation. Although LM
        returns the absolute angle between CAB and DCE as the result,
        it must be noted that intuitively what we are measuring is
        relative change in the angle of second plane (DCE) with respect
        to the first plane (CAB). Therefore, angles > 90 degrees should
        be considered as pi - angle.

        i.e. in Example1 tilt = 30deg. and in Example2 tilt = 180-110 = 70deg.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Bif_torque_local.htm
        """

        parent = bif.parent()
        assert parent is not None, "Bifurcation torque is not defined for root"
        idx = parent.branch().origin_id()[0]
        n = parent.branch().attach.node(idx)

        v1, v2 = self._bif_vector_local(n)
        u1, u2 = self._bif_vector_local(bif)

        n1, n2 = np.cross(v1, v2), np.cross(u1, u2)
        theta_deg = np.degrees(angle(n1, n2))
        raise theta_deg

    def bif_torque_remote(self, bif: Tree.Node) -> float:
        """Bifurcation torque.

        This function returns the angle between, current plane of
        bifurcation and previous plane of bifurcation. This is a
        bifurcation level metric and from the figure, the current
        plane of bifurcation is formed between C D and E where all
        three are bifurcation points and the previous plane is formed
        between A B and C bifurcation points.

        A torque is the inner angle measured between two planes
        (current & parent) of bifurcations. Plane DCE has current
        bifurcation and plane CAB has parent bifurcation. Although LM
        returns the absolute angle between CAB and DCE as the result,
        it must be noted that intuitively what we are measuring is
        relative change in the angle of second plane (DCE) with respect
        to the first plane (CAB). Therefore, angles > 90 degrees should
        be considered as pi - angle.

        i.e. in Example1 tilt = 30deg. and in Example2 tilt = 180-110 = 70deg.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Bif_torque_remote.htm
        """

        parent = bif.parent()
        assert parent is not None, "Bifurcation torque is not defined for root"
        idx = parent.branch().origin_id()[0]
        n = parent.branch().attach.node(idx)

        v1, v2 = self._bif_vector_remote(n)
        u1, u2 = self._bif_vector_remote(bif)

        n1, n2 = np.cross(v1, v2), np.cross(u1, u2)
        theta_deg = np.degrees(angle(n1, n2))
        raise theta_deg

    def _bif_vector_local(
        self, bif: Tree.Node
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        children = bif.children()
        assert len(children) == 2, "Only defined for bifurcations"

        v1 = children[0].xyz() - bif.xyz()
        v2 = children[1].xyz() - bif.xyz()
        return v1, v2

    def _bif_vector_remote(
        self, bif: Tree.Node
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        children = bif.children()
        assert len(children) == 2, "Only defined for bifurcations"

        v1 = children[0].branch()[-1].xyz() - bif.xyz()
        v2 = children[1].branch()[-1].xyz() - bif.xyz()
        return v1, v2

    def last_parent_diam(self, branch: Branch) -> float:
        """The diameter of last bifurcation before the terminal tips.

        This function returns the diameter of last bifurcation before
        the terminal tips.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Last_parent_diam.htm
        """

        raise NotImplementedError()

    def diam_threshold(self, branch: Branch) -> float:
        """

        This function returns Diameter of first compartment after the
        last bifurcation leading to a terminal tip.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Diam_threshold.htm
        """

        raise NotImplementedError()

    def hillman_threshold(self, branch: Branch) -> float:
        """Hillman Threshold.

        Computes the weighted average between 50% of father and 25% of
        daughter diameters of the terminal bifurcation.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/HillmanThreshold.htm
        """

        raise NotImplementedError()

    # Compartment level geometrical measurements

    def diameter(self, node: Node) -> float:
        """This function returns diameter of each compartment the neuron.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Diameter.htm
        """

        return 2 * node.r

    def diameter_pow(self, node: Node) -> float:
        """Computes the diameter raised to the power 1.5 for each compartment.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Diameter_pow.htm
        """

        return self.diameter(node) ** 1.5

    def length(self, compartment: Compartment) -> float:
        """Length of the compartment.

        This function returns the length of compartments by computing
        the distance between the two end points of a compartment.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Length.htm
        """
        return compartment.length()

    def surface(self, compartment: Compartment) -> float:
        """This function returns surface of the compartment.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Surface.htm
        """
        p = compartment[self.compartment_point]
        return cylinder_side_surface_area(p.r, compartment.length())

    def section_area(self, node: Node) -> float:
        """This function returns the SectionArea of the compartment.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/SectionArea.htm
        """
        return circle_area(node.r)

    def volume(self, compartment: Compartment) -> float:
        """This function returns the volume of the compartment.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Volume.htm
        """
        p = compartment[self.compartment_point]
        return cylinder_volume(p.r, compartment.length())

    def euc_distance(self, node: Tree.Node) -> float:
        """Euclidean distance from compartment to soma.

        This function returns the Euclidean distance of a compartment
        with respect to soma.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/EucDistance.htm
        """

        soma = node.attach.soma()
        return node.distance(soma)

    def path_distance(self, node: Tree.Node) -> float:
        """Path distance from compartment to soma.

        This function returns the PathDistance of a compartment.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/PathDistance.htm
        """

        n = node
        length = 0
        while (parent := n.parent()) is not None:
            length += n.distance(parent)
            n = parent
        return length

    def branch_order(self, node: Tree.Node) -> int:
        """The order of the compartment.

        This function returns the order of the branch with respect to soma.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Branch_Order.htm
        """

        n = node
        order = 0
        while n is not None:
            if n.is_bifurcation():
                order += 1
            n = n.parent()
        return order

    def terminal_degree(self, node: Tree.Node) -> int:
        """The number of tips of the comparment.

        This function gives the total number of tips that each
        compartment will terminate into.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Terminal_degree.htm
        """

        return len(node.subtree().get_tips())

    def helix(self, compartment: Tree.Compartment) -> float:
        """Helix of the compartment.

        The function computes the helix by choosing the 3 segments at a
        time (or four points at a time) and computes the normal form on
        the 3 vectors to find the 4th vector.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Helix.htm
        """

        n1 = compartment.attach.node(compartment.origin_id()[0])
        n2 = compartment.attach.node(compartment.origin_id()[1])
        parent = n1.parent()
        assert parent is not None
        grandparent = parent.parent()
        assert grandparent is not None

        a = n1.xyz() - parent.xyz()
        b = parent.xyz() - grandparent.xyz()
        c = n2.xyz() - n1.xyz()
        return np.dot(np.cross(a, b), c) / (
            3 * np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)
        )

    # Whole-arbor measurements

    def width(self, tree: Tree) -> float:
        """With of the neuron.

        Width is computed on the x-coordinates and it is the difference
        of minimum and maximum x-values after eliminating the outer
        points on the either ends by using the 95% approximation of the
        x-values of the given input neuron.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Width.htm
        """
        return max_filtered_difference(tree.x(), 95)

    def height(self, tree: Tree) -> float:
        """Height of the neuron.

        Height is computed on the y-coordinates and it is the
        difference of minimum and maximum y-values after eliminating
        the outer points on the either ends by using the 95%
        approximation of the y-values of the given input neuron.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Height.htm
        """
        return max_filtered_difference(tree.y(), 95)

    def depth(self, tree: Tree) -> float:
        """Depth of the neuron.

        Depth is computed on the x-coordinates and it is the difference
        of minimum and maximum x-values after eliminating the outer
        points on the either ends by using the 95% approximation of the
        x-values of the given input neuron.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Depth.htm
        """
        return max_filtered_difference(tree.z(), 95)

    # Specific-arbor type measurements

    def soma_surface(self, tree: Tree) -> float:
        """Soma surface area.

        This function computes the surface of the soma (Type=1). If the
        soma is composed of just one compartment, then it uses the
        sphere assumption, otherwise it returns the sum of the external
        cylindrical surfaces of compartments forming the soma. There
        can be multiple soma's, one soma or no soma at all.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Soma_Surface.htm
        """
        return sphere_surface_area(tree.soma().r)  # TODO: handle multiple soma

    def type(self):
        """The type of compartment.

        This function returns type of compartment. Each compartment of
        the neuron is of a particular type; soma = 1, axon = 2, basal
        dendrites = 3, apical dendrites= 4. The type values are
        assigned directly from the given input neuron.

        See Also
        --------
        L-Measure: http://cng.gmu.edu:8080/Lm/help/Type.htm
        """
        raise NotImplementedError()


def max_filtered_difference(
    values: npt.NDArray[np.float32], percentile: float = 95
) -> float:
    assert 0 < percentile < 100, "Percentile must be between 0 and 100"
    sorted = np.sort(values)
    lower_index = int(len(sorted) * (100 - percentile) / 200)
    upper_index = int(len(sorted) * (1 - (100 - percentile) / 200))
    filtered = sorted[lower_index:upper_index]
    difference = filtered[-1] - filtered[0]
    return difference


def circle_area(r: float) -> float:
    return math.pi * r**2


def sphere_surface_area(r: float):
    return 4 * math.pi * r**2


def cylinder_volume(r: float, h: float) -> float:
    return math.pi * r**2 * h


def cylinder_side_surface_area(r: float, h: float):
    return 2 * math.pi * r * h


def pill_surface_area(ra: float, rb: float, h: float) -> float:
    lateral_area = math.pi * (ra + rb) * math.sqrt((ra - rb) ** 2 + h**2)
    top_hemisphere_area = 2 * math.pi * ra**2
    bottom_hemisphere_area = 2 * math.pi * rb**2
    total_area = lateral_area + top_hemisphere_area + bottom_hemisphere_area
    return total_area
