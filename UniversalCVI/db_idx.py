import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from ._utils import valid_methods, valid_linkages

def DB_idx(x, kmax, kmin=2, method="kmeans", indexlist="all", p=2, q=2, linkage=None, n_init=100):
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
    valid_indexlist = ["all", "DB", "DBs"]
    if indexlist not in valid_indexlist:
        raise ValueError(f"Argument 'indexlist' should be one of {valid_indexlist}")
    if not isinstance(p, (int, float)):
        raise TypeError("Argument 'p' must be numeric")
    if not isinstance(q, (int, float)):
        raise TypeError("Argument 'q' must be numeric")
    if method == "kmeans":
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if linkage is not None:
            raise ValueError("Argument 'linkage' must be None when using K-means")
    if method == "hclust":
        if linkage not in valid_linkages:
            raise ValueError(f"Argument 'linkage' should be one of {valid_linkages}")
        H_model = scipy_linkage(x, method=linkage)

    db = []
    dbs = []

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

        sizecluss = np.bincount(cluss)

        S = np.zeros(k)
        for i in range(k):
            C = sizecluss[i]
            if C > 1:
                cenI = xnew[cluss == i]
                S[i] = (np.sum(np.sqrt(np.sum((x[cluss == i] - cenI) ** 2, axis=1)) ** q) / C) ** (1 / q)
            else:
                S[i] = 0

        m = squareform(pdist(centroid, metric="minkowski", p=p))

        r = np.zeros(k)
        rs = np.zeros(k)

        for i in range(k):
            mask = np.arange(k) != i
            m_row = m[i, mask]
            r[i] = np.max((S[i] + S[mask]) / m_row)
            rs[i] = np.max(S[i] + S[mask]) / np.min(m_row)

        db.append(np.mean(r))
        dbs.append(np.mean(rs))

    DB = pd.DataFrame({"k": range(kmin, kmax + 1), "DB": db})
    DBs = pd.DataFrame({"k": range(kmin, kmax + 1), "DBs": dbs})

    if indexlist == "all":
        return {"DB": DB, "DBs": DBs}
    elif indexlist == "DB":
        return {"DB": DB}
    elif indexlist == "DBs":
        return {"DBs": DBs}