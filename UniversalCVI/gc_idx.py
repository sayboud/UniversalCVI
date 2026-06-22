import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import pdist, squareform
from UniversalCVI.clustering import FCMeans, bic_gm
from ._utils import valid_fuzzy_methods

def GC_idx(x, cmax, cmin=2, indexlist="all", method="FCM", fzm=2, n_init=20, max_iter=100):
    x = np.array(x)
    n = x.shape[0]

    if not isinstance(cmax, int):
        raise TypeError("Argument 'cmax' must be an integer")
    if cmax > n:
        raise ValueError("The maximum number of clusters for consideration should be less than or equal to the number of data points in dataset.")
    if not isinstance(cmin, int):
        raise TypeError("Argument 'cmin' must be an integer")
    if cmin <= 1:
        warnings.warn("The minimum number of clusters for consideration should be more than 1")
    valid_gc = ["all", "GC1", "GC2", "GC3", "GC4"]
    if indexlist not in valid_gc:
        raise ValueError(f"Argument 'indexlist' should be one of {valid_gc}")
    if method not in valid_fuzzy_methods:
        raise ValueError(f"Argument 'method' should be one of {valid_fuzzy_methods}")
    if method == "FCM":
        if fzm <= 1:
            raise ValueError("Argument 'fzm' must be greater than 1")
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if not isinstance(max_iter, int):
            raise TypeError("Argument 'max_iter' must be an integer")

    dist_full = squareform(pdist(x, metric='euclidean'))
    dd = np.triu(dist_full, k=1)
    dd_flat_desc = np.sort(dd.flatten())[::-1] 

    Check_GC1 = indexlist in ["all", "GC1"]
    Check_GC2 = indexlist in ["all", "GC2"]
    Check_GC3 = indexlist in ["all", "GC3"]
    Check_GC4 = indexlist in ["all", "GC4"]

    gc1, gc2, gc3, gc4 = [], [], [], []

    for k in range(cmin, cmax + 1):
        if method == "EM":
            model = bic_gm(x, k=k)
            m = model.predict_proba(x)
        elif method == "FCM":
            model = FCMeans(n_clusters=k, n_init=n_init, m=fzm, max_iter=max_iter)
            model.fit(x)
            m = model.membership_

        mt = m.T                        
        NI = m.sum(axis=0)             
        nws = int(np.floor(np.sum(NI * (NI - 1) / 2)))

        if Check_GC1:
            PD1 = m @ mt              
            G1 = np.sum(PD1 * dd)
            PD1_lower = PD1[np.tril_indices(n, k=-1)] 
            Gmax1 = np.sum(dd_flat_desc[:nws] * np.sort(PD1_lower)[::-1][:nws])
            Gmin1 = np.sum(dd_flat_desc[:nws] * np.sort(PD1_lower)[:nws])
            gc1.append((G1 - Gmin1) / (Gmax1 - Gmin1))

        if Check_GC2 or Check_GC3 or Check_GC4:
            PD2 = np.zeros((n, n))
            PD3 = np.zeros((n, n))
            PD4 = np.zeros((n, n))

            for s in range(n):
                ms = m[s, :]            
                if Check_GC2:
                    PD2[s, :] = np.sum(np.minimum(mt, ms[:, np.newaxis]), axis=0)
                if Check_GC3:
                    PD3[s, :] = np.max(ms[:, np.newaxis] * mt, axis=0)
                if Check_GC4:
                    PD4[s, :] = np.max(np.minimum(mt, ms[:, np.newaxis]), axis=0)

            if Check_GC2:
                G2 = np.sum(PD2 * dd)
                PD2_lower = PD2[np.tril_indices(n, k=-1)]
                Gmax2 = np.sum(dd_flat_desc[:nws] * np.sort(PD2_lower)[::-1][:nws])
                Gmin2 = np.sum(dd_flat_desc[:nws] * np.sort(PD2_lower)[:nws])
                gc2.append((G2 - Gmin2) / (Gmax2 - Gmin2))

            if Check_GC3:
                G3 = np.sum(PD3 * dd)
                PD3_lower = PD3[np.tril_indices(n, k=-1)]
                Gmax3 = np.sum(dd_flat_desc[:nws] * np.sort(PD3_lower)[::-1][:nws])
                Gmin3 = np.sum(dd_flat_desc[:nws] * np.sort(PD3_lower)[:nws])
                gc3.append((G3 - Gmin3) / (Gmax3 - Gmin3))

            if Check_GC4:
                G4 = np.sum(PD4 * dd)
                PD4_lower = PD4[np.tril_indices(n, k=-1)]
                Gmax4 = np.sum(dd_flat_desc[:nws] * np.sort(PD4_lower)[::-1][:nws])
                Gmin4 = np.sum(dd_flat_desc[:nws] * np.sort(PD4_lower)[:nws])
                gc4.append((G4 - Gmin4) / (Gmax4 - Gmin4))

    c_range = range(cmin, cmax + 1)
    GC1 = pd.DataFrame({"c": c_range, "GC1": gc1}) if Check_GC1 else None
    GC2 = pd.DataFrame({"c": c_range, "GC2": gc2}) if Check_GC2 else None
    GC3 = pd.DataFrame({"c": c_range, "GC3": gc3}) if Check_GC3 else None
    GC4 = pd.DataFrame({"c": c_range, "GC4": gc4}) if Check_GC4 else None

    result = {"GC1": GC1, "GC2": GC2, "GC3": GC3, "GC4": GC4}
    if indexlist == "all":
        return result
    return {indexlist: result[indexlist]}