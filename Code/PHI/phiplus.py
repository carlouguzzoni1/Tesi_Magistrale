# phi: https://github.com/ppouyaa/PHI.git
# phiplus:
import hvwfg.wfg as hv
import numpy as np
import numba

@numba.njit()
def dominates(x: np.ndarray, y: np.ndarray) -> bool:
# copy from DESDEO (desdeo_tools.utilities.fast_non_dominated_sorting) (https://github.com/industrial-optimization-group)
# def dominates(x: np.ndarray, y: np.ndarray) -> int:
    """Returns true if x dominates y.

    Args:
        x (np.ndarray): First solution. Should be a 1-D array of numerics.
        y (np.ndarray): Second solution. Should be the same shape as x.

    Returns:
        bool: True if x dominates y, false otherwise.
    """
    dom = False
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
        elif x[i] < y[i]:
            dom = True
    return dom

@numba.njit()
def non_dominated(data: np.ndarray) -> np.ndarray:
    # copy from DESDEO (desdeo_tools.utilities.fast_non_dominated_sorting)
    """Finds the non-dominated front from a population of solutions.

    Args:
        data (np.ndarray): 2-D array of solutions, with each row being a single solution.

    Returns:
        np.ndarray: Boolean array of same length as number of solutions (rows). The value is
            true if corresponding solution is non-dominated. False otherwise
    """

    num_solutions = len(data)
    index = np.zeros(num_solutions, dtype=np.bool_)
    index[0] = True
    for i in range(1, num_solutions):
        index[i] = True
        for j in range(i):
            a = data[i]
            b = data[j]
            if not index[j]:
                continue
            if dominates(data[i], data[j]):
                index[j] = False
            elif dominates(data[j], data[i]):
                index[i] = False
                break
    return index

def check_dominance_relation(nondominated_solutions, rp):
    """Finds the relation between non-dominated solutions and the reference point rp

        Args:
            nondominated_solutions: non-dominated solutions of the population
            rp: reference point

        Returns:
            comp: 1 represents the solution dominates rp, -1 represents the solution is dominated by rp,
                  0 is non-dominated to rp
        """
    comp = np.zeros(len(nondominated_solutions))
    comp_dominates_r = np.ones(len(nondominated_solutions))  # find the position of the solution dominate rp
    comp_dominated_r = np.ones(len(nondominated_solutions))  # find the position of the solution is dominated by rp
    for i in range(len(rp)):
        sol_i = nondominated_solutions[:, i]  # solutions in the ith column
        c1 = (sol_i < rp[i])
        c2 = (sol_i > rp[i])
        comp_dominates_r = np.logical_and(comp_dominates_r, c1)
        comp_dominated_r = np.logical_and(comp_dominated_r, c2)
    if sum(comp_dominates_r) > 0:  # There exists at least one solution that dominates r
        comp = comp + comp_dominates_r
    if sum(comp_dominated_r) > 0:  # There exists at least one solution that is dominated by r
        comp = comp - comp_dominated_r

    return comp

def further_judgment(nondominated_solutions, rp, lp, up):
    """Find solutions in and outside the modified ROI for PHI+, and check if the reference point rp is dominated

        Args:
            nondominated_solutions: non-dominated solutions of the population
            rp: reference point
            lp: lower point
            up: upper point

        Returns:
            flag: 1 represents rp is dominated, 2 represents rp is not dominated
            solIn_phiplus: solutions in the modified ROI
        """
    res = check_dominance_relation(nondominated_solutions, rp)

    check_res = np.isin([1, -1, 0], res)
    solInROI_phiplus = []
    solOutsideROI_phiplus = []

    flag = 0
    comp = np.ones(len(nondominated_solutions))
    for co in range(len(rp)):  # find the soluton in the desirable range
        sol = nondominated_solutions[:, co]
        c1 = (sol >= lp[co])
        c2 = (sol <= up[co])
        comp = np.logical_and(comp, c1)
        comp = np.logical_and(comp, c2)
    if check_res[0]:  # if there is a 1 in res, namely there is a solution that dominates r
        flag = 1
        index_dominates = [res == 1]  # the solution that dominates r
        index_dominates = np.array(index_dominates).flatten()
        comp = np.logical_or(comp, index_dominates)
    else:
        flag = 2
    if sum(comp) > 0:  # There exists at least one solution is within ROI
        solInROI_phiplus = nondominated_solutions[comp]
        solOutsideROI_phiplus = nondominated_solutions[~comp]
    else:
        solOutsideROI_phiplus = nondominated_solutions

    return flag, solInROI_phiplus

class phiplus():
    """Example:
        pplus = phiplus(None, None)
    or
        nadir = np.array([2,2])
        ideal = np.array([0,0])
        pplus = phiplus(nadir, ideal)

        deviations = np.array([0.2,0.3])
        value = pplus.get_phiplus(solutions, rp, deviations, deviations)
    """
    def __init__(self, nadir, ideal):
        """Initialize with an ideal point and a nadir point for the normalization
           If this information is not available, it can be set to None
           If either value is None, the normalization step will be skipped.
        """
        self.nadir = nadir
        self.ideal = ideal

    def normalization(self, s):
        normalized_s = s.copy()
        for i in range(len(self.nadir)):
            min_value = self.ideal[i]
            max_value = self.nadir[i]
            normalized_s[:,i] = (s[:,i] - min_value) / (max_value - min_value)
        return normalized_s


    def get_phiplus(self, set_of_solutions, rp, devations_l, devations_r):
        rp = rp.astype('float64')  # a reference point
        up = rp + devations_r
        lp = rp - devations_l
        up = up.astype('float64')  # an upper point
        lp = lp.astype('float64')  # a lower point

        if len(set_of_solutions) == 0:  # if there is no solution
            return 0
        index_nondominated = non_dominated(set_of_solutions)
        non_dominated_solutions = set_of_solutions[index_nondominated]
        if len(non_dominated_solutions) == 0:  # if there is no non-dominated solutions
            return 0

        flag, sol_In_ROI = further_judgment(non_dominated_solutions, rp, lp, up)

        if len(sol_In_ROI) == 0:  # if there is no solution in the modified ROI
            return 0

        # normalization
        if self.nadir is not None and self.ideal is not None:  # if it is feasible for normalization
            s = self.normalization(sol_In_ROI)
            r = self.normalization(np.asanyarray(rp).reshape(1, -1))
            l = self.normalization(np.asanyarray(lp).reshape(1, -1))
            # set the upper point as the nadir for the calculation of HV
            nadir = self.normalization(np.asanyarray(up).reshape(1, -1))
            nadir = nadir.flatten()
        else:
            s = sol_In_ROI
            r = rp
            l = lp
            nadir = up.flatten()

        # calculation the value of PHI+
        r_hv = hv(np.asanyarray(r).reshape(1, -1), nadir)  # HV value of the normalized reference point r or rp
        l_hv = hv(np.asanyarray(l).reshape(1, -1), nadir)  # HV value of the normalized reference point l or lp
        s_hv = hv(np.asanyarray(s), nadir)  # HV value of normalized solutions or solutions

        if r_hv == 0:  # if r is worse than the upper point
            return 0
        else:
            if flag == 1:  # r is dominated
                results = s_hv/r_hv
            if flag == 2:  # r is not dominated
                dev = r - l  # Left deviations after normalization
                results = s_hv / (l_hv-np.prod(dev))

        return results
