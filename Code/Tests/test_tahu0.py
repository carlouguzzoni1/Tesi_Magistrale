# Test script used to visualize the impact of the choosen method to compute
# \tahu_0 (see the paper) on the nondominated count of solutions as the
# algorithm evolves.

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

from Ellipsoid import Ellipsoid

from R_metric import R_Metric

from PHI import PHI



# --- Parameters. ---
problem_name    = "dtlz1"
n_obj           = 3

ref_dir         = np.array([0.6, 0.3, 0.1])
fair_dir        = 1 / ref_dir

pop_size        = 150
n_gen           = 300

epsilon         = 1.2
# alphas          = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
alphas          = [1e-4, 1e-3, 1e-2, 1e-1]  # Uncomment for shorter runs.
# alphas          = [1e-3, 1e-2]              # Uncomment for shorter runs.
# t0_modes        = ["min", "mean", "trim_mean"]
t0_modes        = ["min", "mean"]           # Uncomment for shorter runs.
# t0_modes        = ["mean"]                  # Uncomment for shorter runs.

window_size     = 10

extent          = 0.2

problem         = get_problem(problem_name, n_obj=n_obj)
ref_dirs        = get_reference_directions("das-dennis", n_obj, n_partitions=14)
pf              = problem.pareto_front(ref_dirs)


# --- Utils. ---

def plot_3d(ax, F, pf, ref_dir, fair_dir, title=None):
    """
    Utility function to plot the pareto front and solutions in a 3d objective space.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        F (np.ndarray): The solutions in the objective space.
        pf (np.ndarray): The pareto front in the objective space.
        ref_dir (np.ndarray): The reference direction.
        fair_dir (np.ndarray): The fairness direction.
        title (str, optional): The title of the plot.
    """
    
    # PF.
    if pf.shape[1] == 3:
        ax.plot_trisurf(
            pf[:, 0], pf[:, 1], pf[:, 2],
            alpha=0.25, color="gray"
        )

    # Reference and fairness lines.
    ax.plot(
        [0, ref_dir[0]],
        [0, ref_dir[1]],
        [0, ref_dir[2]],
        color="orange",
        linewidth=2,
        label="Preference"
    )

    ax.plot(
        [0, fair_dir[0]],
        [0, fair_dir[1]],
        [0, fair_dir[2]],
        color="green",
        linewidth=2,
        label="Fairness"
    )

    # Solutions.
    ax.scatter(
        F[:, 0], F[:, 1], F[:, 2],
        color="red", s=8, label="Solutions"
    )

    # Axis labels.
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")

    # Legend.
    ax.legend(loc="upper right", fontsize=6)

    # Title.
    ax.set_title(title)


def compute_metrics(problem, fair_dir, pf, extent, res):
    """
    Computes R-IGD+ and R-HV for a set of solutions.
    Also computes the PHI metric.
    
    Args:
        problem (pymoo.core.problem.Problem): The problem.
        fair_dir (np.ndarray): The fairness direction.
        pf (np.ndarray): The pareto front.
        extent (float): The extent of the hypercube (see R-metric paper).
        res (pymoo.core.result.Result): The result object containing the solutions.
        
    Returns:
        r_igdp (float): The R-IGD+ metric.
        r_hv (float): The R-HV metric.
        phi (float): The PHI metric.
    """
    
    # Set the reference and worst points.
    z_r = fair_dir / 3
    z_w = fair_dir * 3
    
    # Initialize the R-metric.
    R_solution = R_Metric(z_r=z_r, z_w=z_w, w=fair_dir, extent=extent)
    R_PF = R_Metric(z_r=z_r, z_w=z_w, w=fair_dir, extent=extent)

    # Compute the transferred solutions and PF.
    S_transferred_solution = R_solution.compute(res.F)
    PF_transferred = R_PF.compute(pf)

    # Also transfer the nadir point of the PF to compute R-HV.
    nadir_PF_transferred = np.max(PF_transferred, axis=0)
    
    # Compute R-IGD+.
    r_igdp_solution = IGDPlus(PF_transferred)
    r_igdp = r_igdp_solution(S_transferred_solution)
    
    # Compute R-HV.
    r_hv_solution = HV(ref_point=nadir_PF_transferred)
    r_hv = r_hv_solution.do(S_transferred_solution)
    
    # Compute PHI.
    phi_solution = PHI(nadir=problem.nadir_point())
    phi = phi_solution.get_phi(res.F, RP=z_r)

    return r_igdp, r_hv, phi


# --- Algorithm runs. ---
for t0 in t0_modes:

    # Create a grid of subplots.
    fig, axes = plt.subplots(
        nrows=5,
        ncols=len(alphas),
        figsize=(4 * len(alphas), 20),
        sharex=False
    )
    
    # Edit frames for the 3d plots.
    for col in range(len(alphas)):
        # Default axes are useless as we add a 3d plot with its own axes.
        axes[0, col].axis("off")
        axes[0, col] = fig.add_subplot(5, len(alphas), col + 1, projection="3d")
        axes[1, col].axis("off")
        axes[1, col] = fig.add_subplot(5, len(alphas), col + 1 + len(alphas), projection="3d")

    # If there is only one alpha value, reshape the axes array to plot in a row.
    if len(alphas) == 1:
        axes = np.array(axes).reshape(5, 1)

    for col, alpha in enumerate(alphas):

        # Algorithms.
        algorithm_nosoften = Ellipsoid(
            w=ref_dir,
            alpha=alpha,
            epsilon=epsilon,
            soften_alpha=False,
            t0_mode=t0,
            pop_size=pop_size,
        )

        algorithm_soften = Ellipsoid(
            w=ref_dir,
            alpha=alpha,
            epsilon=epsilon,
            soften_alpha=True,
            t0_mode=t0,
            pop_size=pop_size,
        )

        # Solutions.
        res_nosoften = minimize(
            problem,
            algorithm_nosoften,
            termination=("n_gen", n_gen),
            seed=0,
            verbose=True
        )

        res_soften = minimize(
            problem,
            algorithm_soften,
            termination=("n_gen", n_gen),
            seed=0,
            verbose=True,
            save_history=True
        )

        # --- Data to be used in the plots. ---
        # Nondominated count history.
        nondom_history = res_soften.history[-1].survival.nondom_count_history

        # Statistics.
        mean_val = np.mean(nondom_history)
        median_val = np.median(nondom_history)
        std_val = np.std(nondom_history)
        
        # Moving average.
        ma = pd.Series(nondom_history).rolling(
            window=window_size,
            min_periods=1,
            center=True
            ).mean().values

        # Anomalies.
        anomalies = nondom_history - ma

        # Mean and standard deviation of anomalies.
        mean_anom = np.mean(anomalies)
        std_anom = np.std(anomalies)

        # Normalized directions to show on 3d plots.
        normalized_ref_dir = ref_dir / np.linalg.norm(ref_dir)
        normalized_fair_dir = fair_dir / np.linalg.norm(fair_dir)

        # --- Plots. ---
        # 3d-plots.
        # 1 - No softening.
        ax_3d_nosoften = axes[0, col]

        nosoften_igdp, nosoften_hv, nosoften_phi = compute_metrics(
            problem,
            normalized_fair_dir,
            pf,
            extent,
            res_nosoften)

        ax_3d_nosoften_title = (
            f"alpha = {alpha}, no softening applied\n"
            f"nondom count: {res_nosoften.F.shape[0]}\n"
            f"IGD+: {nosoften_igdp:.6f}\n"
            f"HV: {nosoften_hv:.6f}\n"
            f"PHI: {nosoften_phi:.6f}"
        )

        plot_3d(
            ax_3d_nosoften,
            res_nosoften.F,
            pf,
            normalized_ref_dir,
            normalized_fair_dir,
            ax_3d_nosoften_title
            )

        # 2 - Softening.
        ax_3d_soften = axes[1, col]
        
        soften_igdp, soften_hv, soften_phi = compute_metrics(
            problem,
            normalized_fair_dir,
            pf,
            extent,
            res_soften
            )

        ax_3d_soften_title = (
            f"alpha = {alpha}, softening applied\n"
            f"nondom count: {res_soften.F.shape[0]}\n"
            f"IGD+: {soften_igdp:.6f}\n"
            f"HV: {soften_hv:.6f}\n"
            f"PHI: {soften_phi:.6f}"
        )


        plot_3d(
            ax_3d_soften,
            res_soften.F,
            pf,
            normalized_ref_dir,
            normalized_fair_dir,
            ax_3d_soften_title
            )

        # Line plot.
        ax_line = axes[2, col]
        
        ax_line.plot(nondom_history, linewidth=1.8, label="Nondom count")
        ax_line.plot(ma, linewidth=1.8, color="red", label="Moving average")
        
        ax_line.legend(loc="lower right")

        ax_line.set_ylabel("Nondom count")
        ax_line.set_xlabel("Generation")

        # Box plot.
        ax_box = axes[3, col]
        
        ax_box.boxplot(nondom_history)
        
        ax_box.text(
            0.5,
            1.05,
            f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}",
            transform=ax_box.transAxes,
            ha="center",
            va="bottom"
        )
        
        ax_box.set_ylabel("Nondom count")
        ax_box.set_xticklabels([])

        # Instability plot.
        ax_inst = axes[4, col]
        
        ax_inst.text(
            0.5,
            1.05,
            f"Mean: {mean_anom:.2f}\nStd: {std_anom:.2f}",
            transform=ax_inst.transAxes,
            ha="center",
            va="bottom"
        )
        
        ax_inst.plot(anomalies, linewidth=1.8)
        ax_inst.set_ylabel("Moving average - Nondom count")
        ax_inst.set_xlabel("Generation")

    # Title and layout.
    fig.suptitle(f"Nondom count history.\nt0_mode = \"{t0}\", epsilon = {epsilon}, alpha_decay = {algorithm_soften.alpha_decay}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save.
    fname = f"plots_nondom_history_{t0}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
