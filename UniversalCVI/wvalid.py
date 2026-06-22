import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from scipy.spatial.distance import pdist
from ._utils import valid_methods, valid_linkages, valid_corr, compute_corr

def Wvalid(x, kmax, kmin=2, method="kmeans", corr="pearson", n_init=100, sampling=1, NCstart=True, linkage=None):
    x = np.array(x)
    dm = x.shape

    if not isinstance(kmax, int):
        raise TypeError("Argument 'kmax' must be an integer")
    if kmax > dm[0]:
        raise ValueError("The maximum number of clusters for consideration should be less than or equal to the number of data points in dataset.")
    if not isinstance(kmin, int):
        raise TypeError("Argument 'kmin' must be an integer")
    if method not in valid_methods:
        raise ValueError(f"Argument 'method' should be one of {valid_methods}")
    if corr not in valid_corr:
        raise ValueError(f"Argument 'corr' should be one of {valid_corr}")
    if method == "kmeans":
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if linkage is not None:
            raise ValueError("Argument 'linkage' must be None when using K-means")
    if method == "hclust":
        if linkage not in valid_linkages:
            raise ValueError(f"Argument 'linkage' should be one of {valid_linkages}")
    if not isinstance(sampling, (int, float)):
        raise TypeError("Argument 'sampling' must be numeric")
    if not (0 < sampling <= 1):
        raise ValueError("'sampling' must be greater than 0 and less than or equal to 1")
    if not isinstance(NCstart, bool):
        raise TypeError("Argument 'NCstart' must be a boolean")

    if sampling < 1:
        n_sample = int(np.ceil(dm[0] * sampling))
        sample_idx = np.random.choice(dm[0], n_sample, replace=False)
        x = x[sample_idx]

    d = pdist(x)

    crr = np.zeros(kmax - kmin + 3)

    if NCstart:
        global_mean = x.mean(axis=0)
        dtom = np.sqrt(np.sum((x - global_mean) ** 2, axis=1))
        crr[0] = np.std(dtom, ddof=1) / (np.max(dtom) - np.min(dtom))

    if method == "hclust":
        H_model = scipy_linkage(x, method=linkage)

    lb = 2 if kmin == 2 else kmin - 1

    for k in range(lb, kmax + 2):
        centroid = np.zeros((k, dm[1]))

        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=n_init)
            model.fit(x)
            cluss = model.labels_
            centroid = model.cluster_centers_
            xnew = centroid[cluss]

        elif method == "hclust":
            cluss = cut_tree(H_model, n_clusters=k).flatten()
            for j in range(k):
                pts = x[cluss == j]
                if pts.shape[0] == 1:
                    centroid[j] = pts[0]
                else:
                    centroid[j] = pts.mean(axis=0)
            xnew = centroid[cluss]

        if len(np.unique(cluss)) < k:
            warnings.warn("Some clusters are empty.")

        d2 = pdist(xnew)
        crr[k - kmin + 1] = compute_corr(d, d2, corr)

    K = len(crr)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_forward = (crr[1:K-1] - crr[0:K-2]) / (1 - crr[0:K-2])
        ratio_backward = (crr[2:K] - crr[1:K-1]) / (1 - crr[1:K-1])
        NWI = ratio_forward / np.maximum(0, ratio_backward)

    NWI2 = ratio_forward - ratio_backward
    NWI3 = NWI.copy()

    if np.max(NWI) < np.inf:
        if np.min(NWI) == -np.inf:
            NWI3[NWI == -np.inf] = np.min(NWI[np.isfinite(NWI)])

    if np.max(NWI) == np.inf:
        NWI3[NWI == np.inf] = np.max(NWI[np.isfinite(NWI)]) + NWI2[NWI == np.inf]
        NWI3[NWI < np.inf] = NWI[NWI < np.inf] + NWI2[NWI < np.inf]
        if np.min(NWI) == -np.inf:
            NWI3[NWI == -np.inf] = np.min(NWI[np.isfinite(NWI)]) + NWI2[NWI == -np.inf]

    NC = pd.DataFrame({"k": range(kmin - 1, kmax + 2), "NC": crr})
    
    NCI1 = pd.DataFrame({"k": range(kmin, kmax + 1), "NCI1": NWI})
    NCI2 = pd.DataFrame({"k": range(kmin, kmax + 1), "NCI2": NWI2})
    NCI = pd.DataFrame({"k": range(kmin, kmax + 1), "NCI": NWI3})

    return {"NC": NC, "NCI": NCI, "NCI1": NCI1, "NCI2": NCI2}