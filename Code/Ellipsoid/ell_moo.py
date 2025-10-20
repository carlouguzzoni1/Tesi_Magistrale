import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.misc import has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# ======================================================================
#  Utility: Pairwise distance with metric tensor.
# ======================================================================
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
    D_sq = np.maximum(D_sq, 0)  # numerical stability
    
    return np.sqrt(D_sq)


# ======================================================================
#  Directional Query MOO (Ellipsoidal method).
# ======================================================================
class DQMOO(GeneticAlgorithm):

    def __init__(
        self,
        ref_dir,
        beta=0.5,
        gamma=0,
        adapt_gamma=True,
        pop_size=100,
        sampling=FloatRandomSampling(),
        selection=RandomSelection(),
        crossover=SBX(eta=30, prob=1.0),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
        n_offsprings=None,
        output=MultiObjectiveOutput(),
        **kwargs
    ):
        # Normalize the reference direction.
        self.ref_dir = ref_dir / np.linalg.norm(ref_dir)

        # Ellipsoidal metric parameters.
        self.beta = beta**2  # Anisotropy factor (epsilon in the paper).
        self.gamma = gamma   # Base Coulomb-like repulsion / utility function ratio.
        self.adapt_gamma = adapt_gamma

        # Fairness direction.
        w_b = 1 / self.ref_dir
        # Normalize the fairness direction.
        w_b = w_b / np.linalg.norm(w_b)

        # Prepare the change of basis matrix from the standard basis to the
        # one with first vector w_b.
        e = np.eye(len(w_b))
        T = np.zeros((len(w_b), len(w_b)))
        T[:, 0] = w_b

        # Gram-Schmidt orthogonalization.
        for k in range(1, len(w_b)):
            z_k = np.zeros(len(w_b))
            for i in range(k):
                z_k += ((e[:, k] @ T[:, i]) / (T[:, i] @ T[:, i])) * T[:, i]
            T[:, k] = e[:, k] - z_k

        # Normalize each column.
        for k in range(len(w_b)):
            T[:, k] /= np.linalg.norm(T[:, k])
        
        # NOTE: check GS with other implementations.

        # Build the A matrix. Remember: just one semi-axis is tunable (ie A[0, 0]).
        A = np.eye(len(w_b))
        A[0, 0] = 1 / self.beta
        
        # Build the metric tensor G = Tᵀ A T.
        self.G = T.T @ A @ T
        
        # NOTE: is it the same as self.G = T @ A @ T.T?

        # Define survival strategy.
        survival = kwargs.pop("survival", None)
        if survival is None:
            survival = DQSurvival(
                ref_dir,
                w_b,
                beta=self.beta,
                gamma=self.gamma,
                adapt_gamma=self.adapt_gamma,
                G=self.G
                )

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            output=output,
            **kwargs
        )

    # ------------------------------------------------------------------
    
    def _setup(self, problem, **kwargs):
        """Ensure the termination criterion is set in generations."""
        if isinstance(self.termination, MaximumFunctionCallTermination):
            n_gen = np.ceil((self.termination.n_max_evals - self.pop_size) / self.n_offsprings)
            self.termination = MaximumGenerationTermination(n_gen)

        if not isinstance(self.termination, MaximumGenerationTermination):
            raise Exception("Please use n_gen or n_eval termination for DQMOO!")

    # ------------------------------------------------------------------
    
    def _set_optimum(self, **kwargs):
        """Identify feasible non-dominated solutions."""
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            nds = self.pop[[i for i, p in enumerate(self.pop) if p.get("opt")]]
            self.opt = nds if len(nds) > 0 else self.pop


# ======================================================================
#  Survival Operator for DQMOO.
# ======================================================================
class DQSurvival(Survival):
    """
    Survival strategy guided by ellipsoidal metric and Coulomb-like repulsion.
    """

    def __init__(self, ref_dir, w_b, beta, gamma, adapt_gamma, G):
        super().__init__(filter_infeasible=True)

        self.beta = beta
        self.gamma = gamma
        self.adapt_gamma = adapt_gamma
        self.ref_dir = ref_dir
        self.w_b = w_b
        self.G = G

        self.ideal = np.full(len(ref_dir), np.inf)
        self.nadir = None
        self.gamma_history = []

    # ------------------------------------------------------------------
    
    def _do(self, problem, pop, n_survive, algorithm=None, n_gen=None, n_max_gen=None, **kwargs):
        """Compute survival selection based on metric distance and repulsion."""

        # Get number of generation and maximum number of generations.
        n_gen = n_gen if n_gen is not None else algorithm.n_gen - 1
        n_max_gen = n_max_gen if n_max_gen is not None else algorithm.termination.n_max_gen

        F = pop.get("F")

        # Update the ideal point.
        self.ideal = np.minimum(F.min(axis=0), self.ideal)

        # Compute pairwise distances in the metric space.
        rec_dis = pairwise_distance(F, self.G)
        # Add epsilon to avoid numerical instabilities (every point has a distance 0 to itself).
        rec_dis += np.eye(rec_dis.shape[0]) + np.finfo(np.float32).eps
        # rec_dis refers to the reciprocal of the distances (see the paper).
        rec_dis = 1 / rec_dis
        # Remove the diagonal.
        rec_dis -= np.eye(rec_dis.shape[0])

        # Compute tahu0 (see the paper).
        t0 = np.min(F @ self.w_b)

        # Adapt gamma dynamically if enabled.
        if self.adapt_gamma:
            # NOTE: is beta = 1 / alpha?
            gam = self.gamma * ((t0 ** 2) * np.sqrt(len(self.w_b))) / (F.shape[0] * self.beta)
        else:
            gam = self.gamma
        self.gamma_history.append(gam)

        # Compute directional score: ellipsoidal distance + Coulombic repulsion term.
        dir_ = np.sqrt(np.diag(F @ self.G @ F.T)) + gam * np.sum(rec_dis, axis=0)

        # Select survivors with minimal score (best trade-off).
        survivors = dir_.argsort()[:n_survive]
        for p in pop:
            p.set("opt", False)

        # Mark the selected solutions as optimal.
        selected = pop[survivors]
        fronts, rank = NonDominatedSorting().do(selected.get("F"), return_rank=True, n_stop_if_ranked=n_survive)
        for p in selected[fronts[0]]:
            p.set("opt", True)

        # Update nadir point.
        self.nadir = selected.get("F").max(axis=0)
        
        return selected
