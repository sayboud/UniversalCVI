import numpy as np
import pandas as pd
import warnings
from UniversalCVI.clustering import FCMeans, bic_gm
from ._utils import valid_fuzzy_methods

def KWON2_idx(x, cmax, cmin=2, method="FCM", fzm=2, n_init=20, max_iter=100):
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
    if method not in valid_fuzzy_methods:
        raise ValueError(f"Argument 'method' should be one of {valid_fuzzy_methods}")
    if method == "FCM":
        if fzm <= 1:
            raise ValueError("Argument 'fzm' must be greater than 1")
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if not isinstance(max_iter, int):
            raise TypeError("Argument 'max_iter' must be an integer")

    global_mean = np.mean(x,axis=0)
    kwon2 = []

    for k in range(cmin, cmax + 1):
        if method == "EM":
            model = bic_gm(x,k=k)
            membership = model.predict_proba(x)
            centers = model.means_

        elif method == "FCM":
            model = FCMeans(n_clusters=k, n_init=n_init, m=fzm, max_iter=max_iter)
            model.fit(x)
            membership = model.membership_
            centers = model.cluster_centers_

        w1 = (dm[0] - k + 1) / dm[0]
        w2 = (k / (k - 1)) ** np.sqrt(2)
        w3 = (dm[0] * k) / (dm[0] - k + 1) ** 2

        d2 = np.sum((centers - global_mean) ** 2, axis=1)

        d4 = np.zeros(k)
        for j in range(k):
            center = centers[j]
            sq_dists = np.sum((x - center) ** 2, axis=1)
            d4[j] = (membership[:, j] ** (2 ** np.sqrt(fzm / 2))) @ sq_dists

        d3 = []
        for i in range(k - 1):
            for j in range(i + 1, k):
                d3.append(np.sum((centers[i] - centers[j]) ** 2))
        d3 = np.array(d3)

        numerator = w1 * ((w2 * np.sum(d4)) + (np.sum(d2) / np.max(d2)) + w3)
        denominator = np.min(d3) + 1/k + 1/(k ** (fzm - 1))

        kwon2.append(numerator / denominator)

    KWON2_data = pd.DataFrame({
        "c": range(cmin, cmax + 1),
        "KWON2": kwon2
    })

    return KWON2_data