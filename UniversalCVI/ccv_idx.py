import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import squareform, pdist
from UniversalCVI.clustering import FCMeans, bic_gm
from ._utils import valid_fuzzy_methods, compute_corr

def CCV_idx(x, cmax, cmin=2, indexlist="all", method="FCM", fzm=2, n_init=20, max_iter=100):
    x = np.array(x)
    dm = x.shape

    if not isinstance(cmax, int):
        raise TypeError("Argument 'cmax' must be an integer")
    if cmax > dm[0]:
        raise ValueError("The maximum number of clusters for consideration should be less than or equal to the number of data points in dataset.")
    if not isinstance(cmin, int):
        raise TypeError("Argument 'cmin' must be an integer")
    if cmin <= 1:
        warnings.warn("The minimum number of clusters for consideration should be more than 1")

    valid_indexlist = ["all", "CCVP", "CCVS"]
    if indexlist not in valid_indexlist:
        raise ValueError(f"Argument 'indexlist' should be one of {valid_indexlist}")
    if method not in valid_fuzzy_methods:
        raise ValueError(f"Argument 'method' should be one of {valid_fuzzy_methods}")
    if method == "FCM":
        if fzm <= 1:
            raise ValueError("Argument 'fzm' must be greater than 1")
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if not isinstance(max_iter, int):
            raise TypeError("Argument 'max_iter' must be an integer")

    dist_matrix = squareform(pdist(x, metric='euclidean'))
    distc = dist_matrix.flatten()

    ccvp = []
    ccvs = []

    for k in range(cmin, cmax + 1):
        if method == "EM":
            model = bic_gm(x,k=k)
            membership = model.predict_proba(x)

        elif method == "FCM":
            model = FCMeans(n_clusters=k, n_init=n_init, m=fzm, max_iter=max_iter)
            model.fit(x)
            membership = model.membership_

        uut = membership @ membership.T
        vnew = 1 - (uut / np.max(uut))
        vnew = vnew.flatten()

        if indexlist in ["all", "CCVP"]:
            ccvp.append(compute_corr(distc - np.mean(distc), vnew - np.mean(vnew), "pearson"))

        if indexlist in ["all", "CCVS"]:
            ccvs.append(compute_corr(distc, vnew, "spearman"))

    CCVP = pd.DataFrame({"c": range(cmin, cmax + 1), "CCVP": ccvp}) if indexlist in ["all", "CCVP"] else None
    CCVS = pd.DataFrame({"c": range(cmin, cmax + 1), "CCVS": ccvs}) if indexlist in ["all", "CCVS"] else None

    if indexlist == "all":
        return {"CCVP": CCVP, "CCVS": CCVS}
    elif indexlist == "CCVP":
        return {"CCVP": CCVP}
    elif indexlist == "CCVS":
        return {"CCVS": CCVS}