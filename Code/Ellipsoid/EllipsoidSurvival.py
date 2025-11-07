# ======================================================================
#  Survival Operator for Ellipsoid.
# ======================================================================

import numpy as np

from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting



class EllipsoidSurvival(Survival):
    """
    Survival strategy guided by ellipsoidal metric and Coulomb-like repulsion.
    """

    def __init__(self, w, u_b, epsilon, alpha, adapt_alpha, soften_alpha, G):
        """
        Initialize the survival strategy.
        
        Args:
            w (np.ndarray): The weights of the objectives.
            u_b (np.ndarray): The normalized fairness direction.
            epsilon (float): Anisotropic factor.
            alpha (float): Base Coulomb-like repulsion / utility function ratio.
            adapt_alpha (bool): Whether to adapt alpha.
            soften_alpha (bool): Whether to soften alpha.
            G (np.ndarray): Metric matrix.
        """
        
        super().__init__(filter_infeasible=True)

        # FIXME: w remains unused.
        self.w = w
        self.u_b = u_b
        self.epsilon = epsilon
        self.alpha = alpha
        self.adapt_alpha = adapt_alpha
        self.soften_alpha = soften_alpha
        self.G = G

        # Initially, set the ideal point to infinity and no nadir point.
        self.ideal = np.full(len(w), np.inf)
        self.nadir = None
        
        self.alpha_history = []
        
        # TEST: now testing. If approved, then fully parametrize or create a
        # separate class.
        if soften_alpha:
            self.base_multiplier = 1
            # self.alpha_decay = 1.00025
            self.alpha_decay = 0.99999
            self.alpha_multiplier = self.base_multiplier
            self.nondom_count_history = []
            self.best_nondom_count = 0
            self.patience = 5
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

        # FIXME: why use min instead of mean?
        # Compute tahu0 (see the paper).
        t0 = np.min(F @ self.u_b)

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

        # TEST: now testing. Decide whether to restore original parameters
        # once non-dominated count raises.
        # Soften alpha for the next iteration, if enabled.
        if self.soften_alpha:
            # Append the number of non-dominated solutions in the current front to the history.
            self.nondom_count_history.append(len(fronts[0]))
            
            # If a new max non-dominated count is found, reset stagnation counter.
            if self.nondom_count_history[-1] > self.best_nondom_count:
                self.best_nondom_count = self.nondom_count_history[-1]
                self.stagnation_counter = 0
            else:
                # Else, increment stagnation counter.
                self.stagnation_counter += 1
                
                # If stagnation counter is reached, soften alpha.
                if self.stagnation_counter >= self.patience:
                    self.alpha_multiplier *= self.alpha_decay
                    self.stagnation_counter = 0
                    self.alpha *= self.alpha_multiplier
                    print("Alpha softened!")

        # Update nadir point.
        self.nadir = selected.get("F").max(axis=0)
        
        return selected
    
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
