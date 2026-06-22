import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, cut_tree


def bic_gm(x, k, covariance_types=("spherical", "diag", "tied", "full")):
    if not isinstance(k, int):
        raise TypeError("Argument 'k' must be an integer")
    if k < 1:
        raise ValueError("Argument 'k' must be at least 1")
    if k > x.shape[0]:
        raise ValueError("Argument 'k' must be less than or equal to the number of data points")
    if not isinstance(covariance_types, (list, tuple)):
        raise TypeError("Argument 'covariance_types' must be a list or tuple")
    valid_cov = {"spherical", "diag", "tied", "full"}
    if not all(c in valid_cov for c in covariance_types):
        raise ValueError(f"Argument 'covariance_types' must contain only values from {valid_cov}")
    z = linkage(x, method="ward")
    hc_labels = cut_tree(z, n_clusters=k).flatten()
    means_init = [np.mean(x[hc_labels == i],axis=0) for i in range(k)]

    best_bic = np.inf
    best_model = None

    for cov_type in covariance_types:
        gm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            means_init=means_init,
            n_init=1,          
            init_params="random", 
            random_state=42
        )
        gm.fit(x)
        bic = gm.bic(x)
        if bic < best_bic:
            best_bic = bic
            best_model = gm

    return best_model