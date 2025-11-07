# ======================================================================
#  PHI implementation for MOO.
#  Reference:
#  A Performance Indicator for Interactive Evolutionary Multiobjective
#  Optimization Methods.
#  Pour et al.
# ======================================================================

import numpy as np

from PHI_utils import *

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.util.dominator import Dominator



class PHI():

    def __init__(self, nadir):
        """
        Initialize with a nadir point for hypervolume calculations.
        
        Args:
            nadir (array): The nadir point for hypervolume calculations.
        """
        
        self.nadir = nadir


    def find_RP_doms(self, S, RP):
        """
        Check if the reference point (RP) is dominated by any solution in S.

        Args:
            S (array): An array of solutions.
            RP (array): The reference point to be checked.

        Returns:
            array: An array of solutions that dominate the reference point.
        """

        s_doms_RP = []

        for s in S:
            if dominates(s, RP):
                s_doms_RP.append(s)

        return np.array(s_doms_RP)


    def get_phi(self, S, RP):
        """
        Get the phi values for a set of solutions and a reference point.

        Args:
            S (array): An array of solutions.
            RP (array): The reference point to be checked.

        Returns:
            tuple: A tuple containing the values of phi1, phi2, phi3, and phi4.
        """
        
        # Non-dominated sorting calculator.
        nds = NonDominatedSorting()
        # Hypervolume calculator.
        hv = HV(ref_point=self.nadir)
        
        # Non-dominated solutions (P in the paper).
        nondom_S = nds.do(S, only_non_dominated_front=True)
        nondom_S = S[nondom_S]
        # Solutions dominating RP.
        s_dom_RP = self.find_RP_doms(S, RP)
        # Non-dominated solutions dominating RP (P^{>} in the paper).
        if len(s_dom_RP) > 0:
            nondoms_dom_RP = nds.do(s_dom_RP, only_non_dominated_front=True)
            nondoms_dom_RP = s_dom_RP[nondoms_dom_RP]
        else:
            nondoms_dom_RP = np.array([])
        # Boolean value indicating whether RP is dominated by some non-dominated solution.
        # In the paper: is_RP_dominated = false -> P^{>} = \emptyset.
        is_RP_dominated = len(nondoms_dom_RP) > 0
        
        # Overall HV value of non-dominated solutions (HV(P, z^{dy}) in the paper).
        all_HV = hv.do(nondom_S)
        # HV value between reference and nadir points (HV(\hat{z}, z^{dy}) in the paper).
        RP_HV = hv.do(np.asanyarray(RP).reshape(1, -1))
        
        # Prevent degenerate cases.
        if RP_HV == 0 or all_HV == 0:
            return 0.0
        
        # Compute the hypervolume inside the desired region (v^{<}).
        if is_RP_dominated:
            HV_inside_DR = RP_HV
        else:
            HV_negative = hv.do(np.vstack((nondom_S, RP))) - RP_HV
                # NOTE: there is no point in using the following here:
                # ... - hv.do(np.vstack((nondoms_dom_RP, RP)))
                # because nondoms_dom_RP is certainly empty.

            HV_inside_DR = all_HV - HV_negative

        # Compute the hypervolume outside the desired region (v^{>}).
        if is_RP_dominated:
            HV_outside_DR = hv.do(nondoms_dom_RP) - HV_inside_DR
        else:
            HV_outside_DR = 0
        
        # Compute the PHI metric.
        PHI = HV_inside_DR / RP_HV + HV_outside_DR / all_HV
        
        return PHI
