# ======================================================================
#  R-metric implementation for MOO.
#  Reference:
#  R-Metric: Evaluating the Performance of Preference-Based
#  Evolutionary Multiobjective Optimization Using Reference Points.
#  Ke Li et al.
# ======================================================================

import numpy as np



class R_Metric:
    
    def __init__(self, z_r, z_w, w, extent):
        """
        R-metric initialization.
        
        Args:
            z_r (np.ndarray): Reference point.
            z_w (np.ndarray): Worst point.
            w (np.ndarray): Weight vector.
            extent (float): Hypercube side.
        """
        
        self.z_r = np.asarray(z_r)
        self.z_w = np.asarray(z_w)
        self.w = np.asarray(w)
        self.extent = extent
        
        self.z_p = None             # Pivot point.
        self.S_trimmed = None       # Trimmed set of solutions.
        self.iso_point = None       # Iso-ASF point.
        self.S_transferred = None   # Transferred set of solutions.

    # ------------------------------------------------------------------

    def ASF(self, x):
        """
        Achievement Scalarization Function for a solution s in S.
        
        Args:
            x (np.ndarray): Solution.
        
        Returns:
            float: ASF value.
        """
        
        return np.max((x - self.z_r) / self.w)

    def find_pivot_point(self, S):
        """
        Find pivot point (solution with minimum ASF value).
        
        Args:
            S (np.ndarray): Set of solutions.
        
        Returns:
            np.ndarray: Pivot point.
        """
        
        asf_values = np.array([self.ASF(x) for x in S])
        self.z_p = S[np.argmin(asf_values)]
        
        return self.z_p


    def trim(self, S):
        """
        Trim solutions outside ROI defined by pivot and extent.
        
        Args:
            S (np.ndarray): Set of solutions.
        
        Returns:
            np.ndarray: Trimmed set of solutions.
        """
        
        lower = self.z_p - self.extent / 2
        upper = self.z_p + self.extent / 2
        mask = np.all((S >= lower) & (S <= upper), axis=1)
        self.S_trimmed = S[mask]
        
        return self.S_trimmed


    def iso_ASF_point(self):
        """
        Compute iso-ASF point on reference line.
        
        Returns:
            np.ndarray: Iso-ASF point.
        """
        
        k = np.argmax((self.z_p - self.z_r) / (self.z_w - self.z_r))
        delta = (self.z_p[k] - self.z_r[k]) / (self.z_w[k] - self.z_r[k])
        self.iso_point = self.z_r + delta * (self.z_w - self.z_r)
        
        return self.iso_point


    def transfer(self):
        """
        Transfer trimmed solutions along preferred direction.
        
        Returns:
            np.ndarray: Transferred set of solutions.
        """
        
        direction = self.iso_point - self.z_p
        self.S_transferred = self.S_trimmed + direction
        
        return self.S_transferred

    # ------------------------------------------------------------------

    def compute(self, S):
        """
        Full R-metric pipeline.
        
        Args:
            S (np.ndarray): Set of solutions.
        
        Returns:
            np.ndarray: Transferred set of solutions.
        """
        
        self.find_pivot_point(S)
        self.trim(S)
        self.iso_ASF_point()
        
        return self.transfer()