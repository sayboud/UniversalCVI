import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from scipy.spatial.distance import pdist
from ._utils import valid_methods, valid_linkages

def STRPBM_idx(x, kmax, kmin=2, method="kmeans", indexlist="all", linkage=None, n_init=100):
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
    valid_indexlist = ["all", "STR", "PBM"]
    if indexlist not in valid_indexlist:
        raise ValueError(f"Argument 'indexlist' should be one of {valid_indexlist}")
    if method == "kmeans":
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if linkage is not None:
            raise ValueError("Argument 'linkage' must be None when using K-means")
    if method == "hclust":
        if linkage not in valid_linkages:
            raise ValueError(f"Argument 'linkage' should be one of {valid_linkages}")
        H_model = scipy_linkage(x, method=linkage)

    EK = np.zeros(kmax - kmin + 3)
    DK = np.zeros(kmax - kmin + 4)
    md = np.zeros(kmax - kmin + 4)

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

        EK[k - kmin + 1] = np.sum(np.sqrt(np.sum((x - xnew) ** 2, axis=1)))
        ddd = pdist(centroid)
        md[k - kmin + 1] = np.max(ddd)
        DK[k - kmin + 1] = np.max(ddd) / np.min(ddd)

    E0 = np.sum(np.sqrt(np.sum((x - x.mean(axis=0)) ** 2, axis=1)))

    if kmin == 2:
        EK[0] = E0

    EKK = E0 / EK

    str_vals = (EKK[1:len(EKK)-1] - EKK[0:len(EKK)-2]) * \
           (DK[2:len(DK)-1] - DK[1:len(DK)-2])

    pbm = EKK[1:len(EKK)-1] * md[1:len(EKK)-1] / np.arange(kmin, kmax + 1)

    STR = pd.DataFrame({"k": range(kmin, kmax + 1), "STR": str_vals})
    PBM = pd.DataFrame({"k": range(kmin, kmax + 1), "PBM": pbm})

    if indexlist == "all":
        return {"STR": STR, "PBM": PBM}
    elif indexlist == "STR":
        return {"STR": STR}
    elif indexlist == "PBM":
        return {"PBM": PBM}