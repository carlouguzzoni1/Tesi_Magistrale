"""
This code implements the PHI (Preference-based Hypervolume Indicator) and related decision assessment
methods as introduced in the paper "A Performance Indicator for Interactive Evolutionary Multiobjective 
Optimization Methods." It's designed for analyzing multiobjective optimization problems, taking into 
account decision-maker preferences. The PHI indicator evaluates the performance of solutions relative
to a reference point, focusing on the coverage of the desired solution region.
To run the code to get the phi values you should run get_phi(),and for the decision phase you should
run assess_decision_phase()

For inquiries or further details, contact pouya(dot)aghaeipour(at)gmail.com.
When using this code or its methodology in academic or research work, 
please cite the paper appropriately to acknowledge the original work and its contributors.
P. Aghaei Pour, S. Bandaru, B. Afsar, M. Emmerich and K. Miettinen, "A Performance Indicator
for Interactive Evolutionary Multiobjective Optimization Methods," in IEEE Transactions
on Evolutionary Computation, doi: 10.1109/TEVC.2023.3272953.
"""

# NOTE: modificato in data 19/10/25 per aderire in modo pedissequo al paper.



import numpy as np

from desdeo_tools.utilities.fast_non_dominated_sorting import dominates, fast_non_dominated_sort_indices
from desdeo_tools.utilities.quality_indicator import hypervolume_indicator



class phi():

    def __init__(self, ideal, nadir):
        """
        Initialize with an ideal point for hypervolume calculations.
        
        Args:
            ideal (array): The ideal point for hypervolume calculations.
        """
        
        self.ideal = ideal
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
        
        # Non-dominated solutions (P in the paper).
        nondom_S = fast_non_dominated_sort_indices(S)[0][0]        
        # Solutions dominating RP.
        s_dom_RP = self.find_RP_doms(S, RP)
        # Non-dominated solutions dominating RP (P^{>} in the paper).
        nondoms_dom_RP = fast_non_dominated_sort_indices(s_dom_RP)[0][0]
        # Boolean value indicating whether RP is dominated by some non-dominated solution.
        # In the paper: is_RP_dominated = false -> P^{>} = \emptyset.
        is_RP_dominated = len(nondoms_dom_RP) > 0
        
        # Overall HV value of non-dominated solutions (HV(P, z^{dy}) in the paper).
        all_HV = hypervolume_indicator(nondom_S, self.nadir)
        # HV value between reference and nadir points (HV(\hat{z}, z^{dy}) in the paper).
        RP_HV = hypervolume_indicator(np.asanyarray(RP).reshape(1, -1), self.nadir)
        
        # Prevent degenerate cases.
        if RP_HV == 0 or all_HV == 0:
            return 0.0
        
        # Compute the hypervolume inside the desired region (v^{<}).
        if is_RP_dominated:
            HV_inside_DR = RP_HV
        else:
            HV_negative = hypervolume_indicator(np.vstack((nondom_S, RP)), self.nadir) - \
                hypervolume_indicator(np.vstack((nondoms_dom_RP, RP)), self.nadir)

            HV_inside_DR = all_HV - HV_negative

        # Compute the hypervolume outside the desired region (v^{>}).
        if is_RP_dominated:
            HV_outside_DR = hypervolume_indicator(nondoms_dom_RP, self.nadir) - HV_inside_DR
        else:
            HV_outside_DR = 0
        
        # Compute the PHI metric.
        PHI = HV_inside_DR / RP_HV + HV_outside_DR / all_HV
        
        return PHI
