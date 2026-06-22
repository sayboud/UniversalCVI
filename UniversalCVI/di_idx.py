import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from ._utils import valid_methods, valid_linkages

def DI_idx(x, kmax, kmin=2, method="kmeans", linkage=None, n_init=100):
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
    if method == "kmeans":
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if linkage is not None:
            raise ValueError("Argument 'linkage' must be None when using K-means")
    if method == "hclust":
        if linkage not in valid_linkages:
            raise ValueError(f"Argument 'linkage' should be one of {valid_linkages}")
        H_model = scipy_linkage(x, method=linkage)

    di = []

    for k in range(kmin, kmax + 1):

        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=n_init)
            cluss = model.fit_predict(x)

        elif method == "hclust":
            cluss = cut_tree(H_model, n_clusters=k).flatten()

        if len(np.unique(cluss)) < k:
            warnings.warn("Some clusters are empty.")

        size = np.bincount(cluss)

        dunn_dem = 0
        dunn_num = 1e+10

        for i in range(k - 1):
            pts_i = x[cluss == i]
            if pts_i.shape[0] > 1:
                dunn_dem = max(dunn_dem, np.max(pdist(pts_i)))
            
            for j in range(i + 1, k):
                pts_j = x[cluss == j]
                stacked = np.vstack([pts_i, pts_j])
                dist_matrix = squareform(pdist(stacked))
                inter_dists = dist_matrix[:size[i], size[i]:]
                dunn_num = min(dunn_num, np.min(inter_dists))

        pts_k = x[cluss == k - 1]
        if pts_k.shape[0] > 1:
            dunn_dem = max(dunn_dem, np.max(pdist(pts_k)))

        di.append(dunn_num / dunn_dem)

    DI_data = pd.DataFrame({
        "k": range(kmin, kmax + 1),
        "DI": di
    })

    return DI_data