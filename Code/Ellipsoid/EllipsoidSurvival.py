# ======================================================================
#  Survival Operator for Ellipsoid.
# ======================================================================

import numpy as np

from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from scipy.stats import trim_mean



class EllipsoidSurvival(Survival):
    """
    Survival strategy guided by ellipsoidal metric and Coulomb-like repulsion.
    """

    def __init__(
        self,
        u_b,
        G,
        alpha,
        epsilon,
        adapt_alpha=True,
        soften_alpha=False,
        alpha_decay=0.9,
        patience=5,
        t0_mode="mean"
        ):
        """
        Initialize the survival strategy.
        
        Args:
            u_b (np.ndarray): The normalized fairness direction.
            G (np.ndarray): Metric matrix.
            alpha (float): Base Coulomb-like repulsion / utility function ratio.
            epsilon (float): Anisotropic factor.
            adapt_alpha (bool, optional): Whether to adapt alpha. Defaults to True.
            soften_alpha (bool, optional): Whether to soften alpha. Defaults to True.
            alpha_decay (float, optional): Decay factor for alpha. Defaults to 0.99.
            patience (int, optional): Patience for alpha adaptation. Defaults to 5.
            t0_mode (str, optional): Mode for t0 computation. Defaults to "mean".
        """
        
        super().__init__(filter_infeasible=True)

        self.u_b = u_b
        self.G = G
        self.alpha = alpha
        self.epsilon = epsilon
        self.adapt_alpha = adapt_alpha
        self.soften_alpha = soften_alpha
        self.t0_mode = t0_mode

        # Initially, set the ideal point to infinity and no nadir point.
        self.ideal = np.full(len(u_b), np.inf)
        self.nadir = None
        
        self.alpha_history = []
        
        # TEST: now testing. If approved, then fully parametrize or create a
        # separate class.
        if soften_alpha:
            # self.base_multiplier = 1
            self.alpha_decay = alpha_decay
            # TEST: trying with a specular value that is not equal to the decay.
            self.alpha_recover = 1 + (1 - self.alpha_decay)
            # self.alpha_multiplier = self.base_multiplier
            self.nondom_count_history = []
            self.best_nondom_count = 0
            self.patience = patience
            self.stagnation_counter = 0

    # ------------------------------------------------------------------
    
    def _do(self, problem, pop, n_survive, algorithm=None, n_gen=None, n_max_gen=None, **kwargs):
        """
        Compute survival selection based on metric distance and repulsion.
        
        Args:
            problem (Problem): The optimization problem.
            pop (Population): The current population.
            n_survive (int): The number of survivors to select.
            algorithm (Algorithm): The algorithm.
            n_gen (int): The current generation.
            n_max_gen (int): The maximum number of generations.
            **kwargs: Additional keyword arguments.
        """

        # Get number of generation and maximum number of generations.
        n_gen = n_gen if n_gen is not None else algorithm.n_gen - 1
        n_max_gen = n_max_gen if n_max_gen is not None else algorithm.termination.n_max_gen

        # Get the current population.
        F = pop.get("F")

        # Update the ideal point.
        self.ideal = np.minimum(F.min(axis=0), self.ideal)

        # Compute pairwise distances in the metric space.
        rec_dis = self.pairwise_distance(F, self.G)
        # Add epsilon to avoid numerical instabilities (every point has a distance 0 to itself).
        rec_dis += np.eye(rec_dis.shape[0]) + np.finfo(np.float32).eps
        # Actually compute rec_dis, which refers to the reciprocal of the distances (see the paper).
        rec_dis = 1 / rec_dis
        # Remove the diagonal.
        rec_dis -= np.eye(rec_dis.shape[0])

        # Compute tahu0 (see the paper). Here, different ways are available.
        if self.t0_mode == "min":
            t0 = np.min(F @ self.u_b)
        elif self.t0_mode == "mean":
            t0 = np.mean(F @ self.u_b)
        elif self.t0_mode == "trim_mean":
            t0 = trim_mean(F @ self.u_b, proportiontocut=0.1)

        # Adapt alpha dynamically, if enabled.
        if self.adapt_alpha:
            # NOTE: here, epsilon was previously used as epsilon^2 due to instruction:
            # self.epsilon = epsilon**2 in Ellipsoid.__init__(). Fixed.
            alpha = self.alpha * ((t0 ** 2) * np.sqrt(len(self.u_b))) / (F.shape[0] * self.epsilon)
        else:
            alpha = self.alpha

        # Update alpha history.
        self.alpha_history.append(alpha)

        # Compute directional score (utility function):
        # ellipsoidal distance + Coulombic repulsion term.
        dir_scores = np.sqrt(np.diag(F @ self.G @ F.T)) + alpha * np.sum(rec_dis, axis=0)

        # FIXME: is n_survive = len(pop) / 2?
        # Select survivors with minimal score (best trade-off).
        survivors = dir_scores.argsort()[:n_survive]

        # Initially, set all solutions as not optimal.
        for p in pop:
            p.set("opt", False)

        # Mark the non-dominated ones among survivor solutions as optimal.
        selected = pop[survivors]
        fronts, rank = NonDominatedSorting().do(selected.get("F"), return_rank=True, n_stop_if_ranked=n_survive)

        for p in selected[fronts[0]]:
            p.set("opt", True)

        if self.soften_alpha:
            self._soften_alpha(algorithm, fronts, pop)

        # Update nadir point.
        self.nadir = selected.get("F").max(axis=0)
        
        return selected
    
    # ------------------------------------------------------------------
    
    def _soften_alpha(self, algorithm, fronts, pop):
        # If it is the first generation, don't append the number of non-dominated
        # solutions, as at initialization it is equal to the maximum.
        # Return, instead.
        if algorithm.n_gen == 1:
            return
        
        # Append the number of non-dominated solutions to the history.
        nondom_count = len(fronts[0])
        self.nondom_count_history.append(nondom_count)
        
        # If there was a previous generation (ie it is possible to compare the
        # number of non-dominated solutions).
        if algorithm.n_gen > 2:
            # Get the number of non-dominated solutions of the previous generation.
            prev_nondom_count = self.nondom_count_history[-2]
        else:
            # Else, just return.
            return
        
        # Set actual population size (pop is the population before the selection).
        pop_size = len(pop) / 2
        
        # Compute the ratio between non-dominated solutions and the population size.
        ratio = nondom_count / pop_size
        
        # If the ratio is less than a consistent part of the population (ie. 0.95),
        # soften alpha.
        if ratio > 0.9:
            # Reset the stagnation counter.
            self.stagnation_counter = 0
            # Increment alpha.
            self.alpha *= self.alpha_recover
            print("SOFTENING - Alpha recovered!")
        # Else, if the non-dominated count is less than or equal to the previous one.
        elif nondom_count <= self.best_nondom_count:
            # Increment stagnation counter.
            self.stagnation_counter += 1
            
            # If patience is reached.
            if self.stagnation_counter >= self.patience:
                # Soften alpha.
                self.alpha *= self.alpha_decay
                # Reset stagnation counter.
                self.stagnation_counter = 0
                print("SOFTENING - Alpha softened!")
        # Else, ratio is below goal but the non-dominated count is not decreasing.
        else:
            self.best_nondom_count = nondom_count
            self.stagnation_counter = 0
        
        """
        # Apply softening until a consistent part (95%) of the population is non-dominated.
        if np.max(self.nondom_count_history) < 0.95 * pop_size:
            # If a new max non-dominated count is found, reset stagnation counter.
            if self.nondom_count_history[-1] > self.best_nondom_count:
                self.best_nondom_count = self.nondom_count_history[-1]
                self.stagnation_counter = 0
            else:
                # Else, increment stagnation counter.
                self.stagnation_counter += 1
                
                # If stagnation counter is reached, soften alpha.
                if self.stagnation_counter >= self.patience:
                    # self.alpha_multiplier *= self.alpha_decay
                    self.stagnation_counter = 0
                    self.alpha *= self.alpha_decay
                    print("Alpha softened!")
        """
    
    # ------------------------------------------------------------------
    
    @staticmethod
    def pairwise_distance(X, G):
        """
        Compute pairwise distances under a metric tensor G.

        Parameters
        ----------
        X : np.ndarray (n, d)
            Input points in the objective space.
        G : np.ndarray (d, d)
            Metric tensor.

        Returns
        -------
        D : np.ndarray (n, n)
            Symmetric matrix of pairwise distances.
        """
        
        XG = X @ G
        D_sq = np.sum(XG * X, axis=1, keepdims=True) - 2 * (XG @ X.T) + np.sum(X @ G * X, axis=1)
        D_sq = np.maximum(D_sq, 0)  # For numerical stability.
        
        return np.sqrt(D_sq)
