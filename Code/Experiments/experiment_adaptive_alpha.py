import numpy as np

import pandas as pd

from Ellipsoid import Ellipsoid

from PHI import PHI

from R_metric import R_Metric

from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions



# --- Parameters. ---

# Problem config.
problems        = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]
n_objs          = [3, 5, 10]

# Algorithm config.
ref_dir         = None
pop_size        = 100
epsilon         = 3
alpha           = 0.1
adapt_alpha     = True
soften_alpha    = [False, True]

# Solver config.
n_gen           = 250
seed            = 0
verbose         = True

# Experiment config.
n_runs          = 10

# Das-Dennis config.
n_partitions    = 14

# PF sampling config.
n_objs_thresh   = 5
n_samples       = 10000

# R-metric config.
ref_dir_scale   = 3
extent          = 0.2

# Run experiments.
for value in soften_alpha:
    print(f"Running experiments for alpha: {value}")
    
    for problem_name in problems:
        print(f"Running benchmark problem: {problem_name}")
        
        for n_obj in n_objs:
            print(f"Running with {n_obj} objectives")
            
            # --- Problem initialization and solving. ---
            
            # FIXME: use any particular reference direction? For now it's just uniform.
            # Define the reference direction.
            ref_dir = np.array([1 / n_obj] * n_obj)
            fair_dir = 1 / ref_dir
            
            # Define metrics data structure.
            results = []
            
            for i in range(n_runs):
                # Define the problem to solve.
                problem = get_problem(problem_name, n_obj)
                
                # Initialize the algorithm.
                algorithm = Ellipsoid(
                    ref_dir=ref_dir,
                    pop_size=pop_size,
                    epsilon=epsilon,
                    alpha=alpha,
                    adapt_alpha=adapt_alpha,
                    soften_alpha=value
                    )
                
                # Solve the problem with the algorithm.
                res = minimize(
                    problem,
                    algorithm,
                    termination=('n_gen', n_gen),
                    seed=seed,
                    verbose=verbose
                    )
                
                # TODO: normalize.
                
                # --- Metrics computation. ---
                
                # NOTE: Method for retrieving reference directions can be changed at
                # any time. See: https://pymoo.org/misc/reference_directions.html.
                # Get reference directions with Das-Dennis.
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
                # Compute the Pareto front.
                pf = problem.pareto_front(ref_dirs)
                
                # If the PF is too large (Das-Dennis sampling tends to explode), subsample.
                if n_objs > n_objs_thresh:
                    idx = np.random.choice(pf.shape[0], n_samples, replace=False)
                    pf = pf[idx]
                
                # Compute HV indicator WRT the problem's nadir point.
                hv = HV(ref_point=problem.nadir_point())
                hv_value = hv.do(pf)
                
                # Compute IGD+ metric.
                igdp = IGDPlus(pf)
                igdp_value = igdp(res.F)
                
                # NOTE: in the original notebook, z_r and z_w were set to fair_dir / k
                # and fair_dir * k, respectively. So all on the same line.
                # Compute R-HV and R-IGD+.
                r_Ell = R_Metric(
                    z_r=fair_dir / ref_dir_scale,
                    z_w=fair_dir * ref_dir_scale,
                    w=fair_dir,
                    extent=extent
                )
                
                r_PF = R_Metric(
                    z_r=fair_dir / ref_dir_scale,
                    z_w=fair_dir * ref_dir_scale,
                    w=fair_dir,
                    extent=extent
                )
                
                S_transferred_Ell = r_Ell.compute(res.F)
                S_transferred_PF = r_PF.compute(pf)
                
                # Compute the nadir point of the transferred PF.
                nadir_transferred_PF = np.max(S_transferred_PF, axis=0)
                # Use it as the reference point for R-HV.
                r_hv = HV(ref_point=nadir_transferred_PF)
                r_Hv_value = r_hv.do(S_transferred_Ell)
                
                r_igdp = IGDPlus(S_transferred_PF)
                r_igdp_value = r_igdp.do(S_transferred_Ell)
                
                # FIXME: what is RP in this case? Can the mean of PF be used?
                # Compute PHI.
                phi = PHI(nadir=problem.nadir_point())
                phi_value = phi.get_phi(res.F, RP=None)
                
                # Append the new found results to the data structure.
                results.append(
                    {
                        "problem": problem_name,
                        "n_obj": n_obj,
                        "soften_alpha": value,
                        "hv": hv_value,
                        "igdp": igdp_value,
                        "r_hv": r_Hv_value,
                        "r_igdp": r_igdp_value,
                        "phi": phi_value
                    }
                )
            
            # --- Dumping. ---
            
            df = pd.DataFrame(results)
            df.to_csv("../Results/adaptive_alpha.csv", mode="a", index=False)