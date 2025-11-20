# ======================================================================
#  Ellipsoid MOO method.
# ======================================================================

import numpy as np

from EllipsoidSurvival import EllipsoidSurvival
from Ellipsoid_utils import qr_basis

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.misc import has_feasible



class Ellipsoid(GeneticAlgorithm):
    """
    MOO algorithm designed for many-objective optimization problems.

    The algorithm is based on a particular ellipsoidal metric and simulated
    Coulomb-like repulsion between solutions.
    
    While this class "pre-processes" the problem parameters, the actual
    selection method is implemented in EllipsoidSurvival.
    """

    def __init__(
        self,
        w,
        alpha=0.1,
        epsilon=2,
        adapt_alpha=True,
        soften_alpha=False,
        alpha_decay=0.9,
        patience=5,
        t0_mode="mean",
        pop_size=100,
        survival=None,
        sampling=FloatRandomSampling(),
        selection=RandomSelection(),
        crossover=SBX(eta=30, prob=1.0),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
        n_offsprings=None,
        output=MultiObjectiveOutput(),
        **kwargs
    ):
        """
        Initialize the algorithm.
        
        Args:
            w (np.ndarray): The weights of the objectives.
            alpha (float, optional): Base Coulomb-like repulsion / utility function ratio. Defaults to 0.1.
            epsilon (float, optional): Anisotropic factor. Defaults to 2.
            adapt_alpha (bool, optional): Whether to adapt alpha. Defaults to True.
            soften_alpha (bool, optional): Whether to soften alpha. Defaults to False.
            alpha_decay (float, optional): Alpha decay factor. Defaults to 0.99.
            patience (int, optional): Patience for alpha adaptation. Defaults to 5.
            t0_mode (str, optional): T0 mode. Defaults to "mean".
            pop_size (int, optional): Population size. Defaults to 100.
            survival (Survival, optional): Survival strategy. Defaults to EllipsoidSurvival().
            sampling (Sampling, optional): Sampling strategy. Defaults to FloatRandomSampling().
            selection (Selection, optional): Selection strategy. Defaults to RandomSelection().
            crossover (Crossover, optional): Crossover strategy. Defaults to SBX(eta=30, prob=1.0).
            mutation (Mutation, optional): Mutation strategy. Defaults to PM(eta=20).
            eliminate_duplicates (bool, optional): Whether to eliminate duplicates. Defaults to True.
            n_offsprings (int, optional): Number of offsprings. Defaults to None.
            output (Output, optional): Output strategy. Defaults to MultiObjectiveOutput().
            **kwargs: Additional keyword arguments.
        """
        
        # Check that the weights are all positive.
        if np.any(w < 0):
            raise ValueError("The weights (w) must be positive.")
        
        # Check that the sum of the weights w is 1 (+- epsilon).
        if np.abs(np.sum(w) - 1) > np.finfo(np.float32).eps:
            raise ValueError(
                "The sum of the weights (w) must be 1. Got %s." % np.sum(w)
            )

        # Ellipsoidal metric parameters.
        self.alpha = alpha          # Base Coulomb-like repulsion / utility function ratio.
        self.adapt_alpha = adapt_alpha
        self.soften_alpha = soften_alpha
        self.epsilon = epsilon      # Anisotropic factor.

        # Normalize the reference direction.
        self.w = w / np.linalg.norm(w)
        
        # Fairness direction.
        w_b = 1 / self.w
        # Normalize the fairness direction to get the preference unit vector.
        u_b = w_b / np.linalg.norm(w_b)

        # NOTE: used an optimized Gram-Schmidt orthogonalization with numpy (QR).
        # Prepare the change of basis matrix from the standard basis to the
        # one with first vector u_b.
        T = qr_basis(u_b)

        # Build the A matrix. Remember: just one semi-axis is tunable (ie. A[0, 0]).
        A = np.eye(len(u_b))
        # NOTE: now we don't set epsilon = epsilon**2 anymore. Else, it would get
        # passed to the survival strategy and make incorrect the utility function.
        A[0, 0] = 1 / self.epsilon**2
        
        # NOTE: G = T.T @ A @ T is not the same as G = T @ A @ T.T.
        # From the paper: G = T.inverse().T * A * T.inverse() <=> G = T * A * T.T,
        # because T is orthogonal (ie. T^{-1} = T.T).
        # G = T.T @ A @ T makes it seem like T was furtherly transposed for some reason.
        # If T @ e_1 = u_b, the original formula is to be used.
        # self.G = T.T @ A @ T
        # Build the metric tensor G.
        self.G = T @ A @ T.T

        # Define survival strategy.
        survival = kwargs.pop("survival", None)
        
        # If no survival strategy is provided, use the default one.
        if survival is None:
            self.alpha_decay = alpha_decay
            self.patience = patience
            self.t0_mode = t0_mode
            
            survival = EllipsoidSurvival(
                u_b=u_b,
                G=self.G,
                alpha=self.alpha,
                epsilon=self.epsilon,
                adapt_alpha=self.adapt_alpha,
                soften_alpha=self.soften_alpha,
                alpha_decay=self.alpha_decay,
                patience=self.patience,
                t0_mode=self.t0_mode
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
        """
        Ensure the termination criterion is set in generations.
        
        Args:
            problem (Problem): The optimization problem.
            **kwargs: Additional keyword arguments.
        """
        
        if isinstance(self.termination, MaximumFunctionCallTermination):
            n_gen = np.ceil((self.termination.n_max_evals - self.pop_size) / self.n_offsprings)
            self.termination = MaximumGenerationTermination(n_gen)

        if not isinstance(self.termination, MaximumGenerationTermination):
            raise Exception("Please use n_gen or n_eval termination for Ellipsoid!")

    # ------------------------------------------------------------------
    
    def _set_optimum(self, **kwargs):
        """
        Identify feasible non-dominated solutions.
        
        Args:
            **kwargs: Additional keyword arguments.
        """
        
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            nds = self.pop[[i for i, p in enumerate(self.pop) if p.get("opt")]]
            self.opt = nds if len(nds) > 0 else self.pop
