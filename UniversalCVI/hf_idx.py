import numpy as np
import pandas as pd
import warnings
from UniversalCVI.clustering import FCMeans, bic_gm
from ._utils import valid_fuzzy_methods

def HF_idx(x, cmax, cmin=2, method="FCM", fzm=2, n_init=20, max_iter=100):
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

    hf = []

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

        d5 = np.zeros(k)
        for j in range(k):
            center = centers[j]
            sq_dists = np.sum((x - center) ** 2, axis=1)
            d5[j] = (membership[:, j] ** fzm) @ sq_dists

        d3 = []
        for i in range(k - 1):
            for j in range(i + 1, k):
                d3.append(np.sum((centers[i] - centers[j]) ** 2))
        d3 = np.array(d3)

        adp = (1 / (k * (k - 1))) * np.sum(d3)

        numerator = np.sum(d5) + 2 * adp
        denominator = (dm[0] / 2 * k) * (np.min(d3) + np.median(d3))

        hf.append(numerator / denominator)

    HF_data = pd.DataFrame({
        "c": range(cmin, cmax + 1),
        "HF": hf
    })

    return HF_data