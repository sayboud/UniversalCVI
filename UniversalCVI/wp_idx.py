import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import pdist
from UniversalCVI.clustering import FCMeans, bic_gm
from ._utils import valid_fuzzy_methods, compute_corr, valid_corr

def WP_idx(x, cmax, cmin=2, corr="pearson", method="FCM", fzm=2,
           gamma=None, sampling=1, n_init=20, max_iter=100, NCstart=True):
    x = np.array(x)

    if not isinstance(cmax, int):
        raise TypeError("Argument 'cmax' must be an integer")
    if cmax > x.shape[0]:
        raise ValueError("The maximum number of clusters for consideration should be less than or equal to the number of data points in dataset.")
    if not isinstance(cmin, int):
        raise TypeError("Argument 'cmin' must be an integer")
    if cmin <= 1:
        warnings.warn("The minimum number of clusters for consideration should be more than 1")
    if method not in valid_fuzzy_methods:
        raise ValueError(f"Argument 'method' should be one of {valid_fuzzy_methods}")
    if corr not in valid_corr:
        raise ValueError(f"Argument 'corr' should be one of {valid_corr}")
    if not isinstance(NCstart, bool):
        raise TypeError("Argument 'NCstart' must be a boolean")
    if method == "FCM":
        if fzm <= 1:
            raise ValueError("Argument 'fzm' must be greater than 1")
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if not isinstance(max_iter, int):
            raise TypeError("Argument 'max_iter' must be an integer")
    if not isinstance(sampling, (int, float)):
        raise TypeError("Argument 'sampling' must be numeric")
    if not (0 < sampling <= 1):
        raise ValueError("'sampling' must be greater than 0 and less than or equal to 1")

    if gamma is None:
        gamma = (fzm ** 2 * 7) / 4

    if sampling < 1:
        n_sample = int(np.ceil(x.shape[0] * sampling))
        sample_idx = np.random.choice(x.shape[0], n_sample, replace=False)
        x = x[sample_idx]

    distx = pdist(x, metric='euclidean')
    crr = np.zeros(cmax - cmin + 3)

    if NCstart and cmin <= 2:
        global_mean = np.mean(x, axis=0)
        dtom = np.sqrt(np.sum((x - global_mean) ** 2, axis=1))
        crr[0] = np.std(dtom, ddof=1) / (np.max(dtom) - np.min(dtom))
    else:
        if method == "EM":
            model = bic_gm(x, k=cmin-1)
            membership = model.predict_proba(x)
            centers = model.means_
        elif method == "FCM":
            model = FCMeans(n_clusters=cmin-1, n_init=n_init, m=fzm, max_iter=max_iter)
            model.fit(x)
            membership = model.membership_
            centers = model.cluster_centers_
        mg = membership ** gamma
        xnew = (mg / np.sum(mg,axis=1, keepdims=True)) @ centers
        crr[0] = compute_corr(distx, pdist(xnew), corr)

    if method == "EM":
        model = bic_gm(x, k=cmax+1)
        membership = model.predict_proba(x)
        centers = model.means_
    elif method == "FCM":
        model = FCMeans(n_clusters=cmax+1, n_init=n_init, m=fzm, max_iter=max_iter)
        model.fit(x)
        membership = model.membership_
        centers = model.cluster_centers_
    mg = membership ** gamma
    xnew = (mg / np.sum(mg,axis=1, keepdims=True)) @ centers
    crr[cmax - cmin + 2] = compute_corr(distx, pdist(xnew), corr)

    for k in range(cmin, cmax + 1):
        if method == "EM":
            model = bic_gm(x, k=k)
            membership = model.predict_proba(x)
            centers = model.means_
        elif method == "FCM":
            model = FCMeans(n_clusters=k, n_init=n_init, m=fzm, max_iter=max_iter)
            model.fit(x)
            membership = model.membership_
            centers = model.cluster_centers_
        mg = membership ** gamma
        xnew = (mg / np.sum(mg, axis=1, keepdims=True)) @ centers
        crr[k - cmin + 1] = compute_corr(distx, pdist(xnew), corr)

    K = len(crr)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_forward  = (crr[1:K-1] - crr[0:K-2]) / (1 - crr[0:K-2])
        ratio_backward = (crr[2:K]   - crr[1:K-1]) / (1 - crr[1:K-1])
        WPI   = ratio_forward / np.maximum(0, ratio_backward)
        WPCI2 = ratio_forward - ratio_backward

    WPCI3 = WPI.copy()
    if np.sum(np.isfinite(WPI)) == 0:
        WPCI3[WPI == np.inf]  = WPCI2[WPI == np.inf]
        WPCI3[WPI == -np.inf] = np.minimum(0, WPCI2[WPI == -np.inf])
    else:
        if np.max(WPI) < np.inf:
            if np.min(WPI) == -np.inf:
                WPCI3[WPI == -np.inf] = np.min(WPI[np.isfinite(WPI)])
        if np.max(WPI) == np.inf:
            WPCI3[WPI == np.inf]  = np.max(WPI[np.isfinite(WPI)]) + WPCI2[WPI == np.inf]
            WPCI3[WPI < np.inf]   = WPI[WPI < np.inf] + WPCI2[WPI < np.inf]
            if np.min(WPI) == -np.inf:
                WPCI3[WPI == -np.inf] = np.min(WPI[np.isfinite(WPI)]) + WPCI2[WPI == -np.inf]

    c_range = range(cmin, cmax + 1)
    WPC   = pd.DataFrame({"c": range(cmin - 1, cmax + 2), "WPC": crr})
    WPCI1 = pd.DataFrame({"c": c_range, "WPI1": WPI})     
    WPCI2 = pd.DataFrame({"c": c_range, "WPCI2": WPCI2})
    WP    = pd.DataFrame({"c": c_range, "WP": WPCI3})

    return {"WPC": WPC, "WP": WP, "WPCI1": WPCI1, "WPCI2": WPCI2}