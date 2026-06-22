import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from ._utils import valid_methods, valid_linkages

def SH_idx(x, kmax, kmin=2, method="kmeans", linkage=None, n_init=100):
    x = np.array(x)
    dm = x.shape
    n = dm[0]

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
    if method == "kmeans":
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if linkage is not None:
            raise ValueError("Argument 'linkage' must be None when using K-means")
    if method == "hclust":
        if linkage not in valid_linkages:
            raise ValueError(f"Argument 'linkage' should be one of {valid_linkages}")
        H_model = scipy_linkage(x, method=linkage)

    sc = []

    for k in range(kmin, kmax + 1):

        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=n_init)
            cluss = model.fit_predict(x)

        elif method == "hclust":
            cluss = cut_tree(H_model, n_clusters=k).flatten()

        if len(np.unique(cluss)) < k:
            warnings.warn("Some clusters are empty.")

        size = np.bincount(cluss)
        sss = np.zeros(n)
        s = 0

        for i in range(k):
            pts_i = x[cluss == i]
            ll = size[i]
            mm = squareform(pdist(pts_i))

            if ll > 1:
                for u in range(ll):
                    mask = np.arange(ll) != u
                    a = np.mean(mm[u, mask])

                    b = np.inf
                    for ii in range(k):
                        if ii == i:
                            continue
                        pts_ii = x[cluss == ii]
                        stacked = np.vstack([pts_i[u], pts_ii])
                        dists = squareform(pdist(stacked))[0, 1:]
                        b = min(b, np.mean(dists))

                    sss[s] = (b - a) / max(a, b)
                    s += 1
            else:
                sss[s] = 0
                s += 1

        sc.append(np.mean(sss))

    SH_data = pd.DataFrame({
        "k": range(kmin, kmax + 1),
        "SH": sc
    })

    return SH_data