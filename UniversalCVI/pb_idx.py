import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from scipy.spatial.distance import pdist
from ._utils import valid_methods, valid_linkages, valid_corr, compute_corr

def PB_idx(x, kmax, kmin=2, method="kmeans", corr="pearson", linkage=None, n_init=100):
    x = np.array(x)
    dm = x.shape

    if not isinstance(kmax, int):
        raise TypeError("Argument 'kmax' must be an integer")
    if kmax > dm[0]:
        raise ValueError("The maximum number of clusters for consideration should be less than or equal to the number of data points in dataset.")
    if not isinstance(kmin, int):
        raise TypeError("Argument 'kmin' must be an integer")
    if kmin <= 1:
        warnings.warn("The minimum number of clusters for consideration should be more than 1")
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
        H_model = scipy_linkage(x, method=linkage)

    d = pdist(x)
    pb = []

    for k in range(kmin, kmax + 1):
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

        d3 = pdist(xnew)
        d3[d3 > 0] = 1

        pb.append(compute_corr(d, d3, corr))

    PB_data = pd.DataFrame({
        "k": range(kmin, kmax + 1),
        "PB": pb
    })

    return PB_data