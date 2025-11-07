import numpy as np

from pymoo.util.ref_dirs import get_reference_directions


def get_local_ref_dirs(ref_dir, method="das-dennis", ndim=3, n_partitions=4, alpha=10):
    
    ref_dirs = get_reference_directions(method, ndim, n_partitions=n_partitions)
    rdd = ref_dirs * 1 / alpha
    ref_dir = ref_dir / np.linalg.norm(ref_dir, 1)
    rdd = rdd + (1 / ndim) * (1 - (1 / alpha))
    rdd = rdd + ref_dir - 1 / ndim
    
    return rdd