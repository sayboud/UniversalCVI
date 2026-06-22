import numpy as np
import pandas as pd
import warnings
from scipy.spatial.distance import pdist, squareform
from UniversalCVI.clustering import FCMeans, bic_gm
from ._utils import valid_fuzzy_methods, compute_corr, valid_corr

def FzzyCVIs(x, cmax, cmin=2, indexlist="all", corr="pearson", method="FCM", fzm=2,
             gamma=None, sampling=1, n_init=20, max_iter=100, NCstart=True):
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

    valid_idx = ["all", "WPC", "WP", "WPCI1", "WPCI2", "XB", "KWON", "KWON2",
                 "TANG", "HF", "WL", "PBM", "KPBM", "CCVP", "CCVS",
                 "GC1", "GC2", "GC3", "GC4"]
    if isinstance(indexlist, str):
        indexlist = [indexlist]
    if not all(idx in valid_idx for idx in indexlist):
        raise ValueError(f"Argument 'indexlist' must be one or more of {valid_idx}")
    if method not in valid_fuzzy_methods:
        raise ValueError(f"Argument 'method' should be one of {valid_fuzzy_methods}")
    if method == "FCM":
        if fzm <= 1:
            raise ValueError("Argument 'fzm' must be greater than 1")
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if not isinstance(max_iter, int):
            raise TypeError("Argument 'max_iter' must be an integer")
    if "all" in indexlist or any(i in indexlist for i in ["WPC", "WP", "WPCI1", "WPCI2"]):
        if corr not in valid_corr:
            raise ValueError(f"Argument 'corr' should be one of {valid_corr}")
        if not isinstance(NCstart, bool):
            raise TypeError("Argument 'NCstart' must be a boolean")
    if not isinstance(sampling, (int, float)):
        raise TypeError("Argument 'sampling' must be numeric")
    if not (0 < sampling <= 1):
        raise ValueError("'sampling' must be greater than 0 and less than or equal to 1")

    if gamma is None:
        gamma = (fzm ** 2 * 7) / 4

    if sampling < 1:
        n_sample = int(np.ceil(n * sampling))
        sample_idx = np.random.choice(n, n_sample, replace=False)
        x = x[sample_idx]
        n = x.shape[0]

    do_wp = "all" in indexlist or any(i in indexlist for i in ["WPC", "WP", "WPCI1", "WPCI2"])
    do_ccv = "all" in indexlist or any(i in indexlist for i in ["CCVP", "CCVS"])
    do_gc = "all" in indexlist or any(i in indexlist for i in ["GC1", "GC2", "GC3", "GC4"])
    do_xb = "all" in indexlist or "XB" in indexlist
    do_kwon = "all" in indexlist or "KWON" in indexlist
    do_kwon2= "all" in indexlist or "KWON2"in indexlist
    do_tang = "all" in indexlist or "TANG" in indexlist
    do_hf = "all" in indexlist or "HF" in indexlist
    do_wl = "all" in indexlist or "WL" in indexlist
    do_pbm = "all" in indexlist or "PBM" in indexlist
    do_kpbm = "all" in indexlist or "KPBM" in indexlist

    do_gc1 = "all" in indexlist or "GC1" in indexlist
    do_gc2 = "all" in indexlist or "GC2" in indexlist
    do_gc3 = "all" in indexlist or "GC3" in indexlist
    do_gc4 = "all" in indexlist or "GC4" in indexlist

    if do_wp or do_ccv or do_gc:
        dist_matrix = squareform(pdist(x, metric='euclidean'))
        distx = pdist(x, metric='euclidean')     
        distc = dist_matrix.flatten()           

    if do_gc:
        dd = np.triu(dist_matrix, k=1)           
        dd_flat_desc = np.sort(dd.flatten())[::-1]

    global_mean = np.mean(x, axis=0)
    d7 = np.sqrt(np.sum((x - global_mean) ** 2, axis=1)) 

    def get_model(k):
        if method == "EM":
            model = bic_gm(x, k=k)
            return model.predict_proba(x), model.means_
        else:
            model = FCMeans(n_clusters=k, n_init=n_init, m=fzm, max_iter=max_iter)
            model.fit(x)
            return model.membership_, model.cluster_centers_

    xb, kwon, kwon2, tang, hf, wl, pbm, kpbm = [], [], [], [], [], [], [], []
    ccvp, ccvs = [], []
    gc1, gc2, gc3, gc4 = [], [], [], []
    crr = np.zeros(cmax - cmin + 3) if do_wp else None

    if do_wp:
        if cmin <= 2:
            dtom = np.sqrt(np.sum((x - global_mean) ** 2, axis=1))
            if NCstart:
                crr[0] = np.std(dtom, ddof=1) / (np.max(dtom) - np.min(dtom))
            else:
                crr[0] = 0.0
        else:
            m_b, c_b = get_model(cmin - 1)
            mg = m_b ** gamma
            xnew = (mg / mg.sum(axis=1, keepdims=True)) @ c_b
            crr[0] = compute_corr(distx, pdist(xnew), corr)

        m_b, c_b = get_model(cmax + 1)
        mg = m_b ** gamma
        xnew = (mg / mg.sum(axis=1, keepdims=True)) @ c_b
        crr[cmax - cmin + 2] = compute_corr(distx, pdist(xnew), corr)

    for k in range(cmin, cmax + 1):
        m, c = get_model(k)
        mt = m.T

        if do_xb or do_kwon or do_kwon2 or do_tang or do_hf or do_wl or do_pbm or do_kpbm:
            d1 = np.zeros(k)
            d4 = np.zeros(k)
            d5 = np.zeros(k)
            d6 = np.zeros(k)
            d8 = np.zeros(k)
            for j in range(k):
                center = c[j]
                sq_dists = np.sum((x - center) ** 2, axis=1)
                dists = np.sqrt(sq_dists)
                d1[j] = (m[:, j] ** 2) @ sq_dists
                d4[j] = (m[:, j] ** (2 ** np.sqrt(fzm/2))) @ sq_dists
                d5[j] = (m[:, j] ** fzm) @ sq_dists
                d6[j] = np.sum(m[:, j])
                d8[j] = m[:, j] @ dists

            d2 = np.sum((c - global_mean) ** 2, axis=1)

            d3, d9 = [], []
            for i in range(k - 1):
                for j in range(i + 1, k):
                    diff_sq = np.sum((c[i] - c[j]) ** 2)
                    d3.append(diff_sq)
                    d9.append(np.sqrt(diff_sq))
            d3 = np.array(d3)
            d9 = np.array(d9)

            adp = (1 / (k * (k - 1))) * np.sum(d3)

            w1 = (n - k + 1) / n
            w2 = (k / (k - 1)) ** np.sqrt(2)
            w3 = (n * k) / (n - k + 1) ** 2

            if do_xb:
                xb.append(np.sum(d1) / (n * np.min(d3)))
            if do_kwon:
                kwon.append((np.sum(d1) + np.mean(d2)) / np.min(d3))
            if do_kwon2:
                num = w1 * ((w2 * np.sum(d4)) + (np.sum(d2) / np.max(d2)) + w3)
                den = np.min(d3) + 1/k + 1/(k ** (fzm - 1))
                kwon2.append(num / den)
            if do_tang:
                tang.append((np.sum(d1) + 2 * adp) / (np.min(d3) + 1/k))
            if do_hf:
                num = np.sum(d5) + 2 * adp
                den = (n / 2 * k) * (np.min(d3) + np.median(d3))
                hf.append(num / den)
            if do_wl:
                wl.append(np.sum(d1 / d6) / (np.min(d3) + np.median(d3)))
            if do_pbm:
                pbm.append(((1/k) * (np.sum(d7) * np.max(d9) / np.sum(d8))) ** 2)
            if do_kpbm:
                kpbm.append(((1/k) * (np.max(d9) / np.sum(d8))) ** 2)

        if do_wp:
            mg = m ** gamma
            xnew = (mg / mg.sum(axis=1, keepdims=True)) @ c
            crr[k - cmin + 1] = compute_corr(distx, pdist(xnew), corr)

        if do_ccv:
            uut  = m @ mt
            vnew = (1 - (uut / np.max(uut))).flatten()
            if do_ccv and ("all" in indexlist or "CCVP" in indexlist):
                ccvp.append(compute_corr(distc - np.mean(distc), vnew - np.mean(vnew), "pearson"))
            if do_ccv and ("all" in indexlist or "CCVS" in indexlist):
                ccvs.append(compute_corr(distc, vnew, "spearman"))

        if do_gc:
            NI = m.sum(axis=0)
            nws = int(np.floor(np.sum(NI * (NI - 1) / 2)))

            if do_gc1:
                PD1 = m @ mt
                G1  = np.sum(PD1 * dd)
                PD1_lower = PD1[np.tril_indices(n, k=-1)]
                Gmax1 = np.sum(dd_flat_desc[:nws] * np.sort(PD1_lower)[::-1][:nws])
                Gmin1 = np.sum(dd_flat_desc[:nws] * np.sort(PD1_lower)[:nws])
                gc1.append((G1 - Gmin1) / (Gmax1 - Gmin1))

            if do_gc2 or do_gc3 or do_gc4:
                PD2 = np.zeros((n, n)) if do_gc2 else None
                PD3 = np.zeros((n, n)) if do_gc3 else None
                PD4 = np.zeros((n, n)) if do_gc4 else None
                for s in range(n):
                    ms = m[s, :]
                    if do_gc2:
                        PD2[s, :] = np.sum(np.minimum(mt, ms[:, np.newaxis]), axis=0)
                    if do_gc3:
                        PD3[s, :] = np.max(ms[:, np.newaxis] * mt, axis=0)
                    if do_gc4:
                        PD4[s, :] = np.max(np.minimum(mt, ms[:, np.newaxis]), axis=0)

                if do_gc2:
                    G2 = np.sum(PD2 * dd)
                    PD2_lower = PD2[np.tril_indices(n, k=-1)]
                    Gmax2 = np.sum(dd_flat_desc[:nws] * np.sort(PD2_lower)[::-1][:nws])
                    Gmin2 = np.sum(dd_flat_desc[:nws] * np.sort(PD2_lower)[:nws])
                    gc2.append((G2 - Gmin2) / (Gmax2 - Gmin2))
                if do_gc3:
                    G3 = np.sum(PD3 * dd)
                    PD3_lower = PD3[np.tril_indices(n, k=-1)]
                    Gmax3 = np.sum(dd_flat_desc[:nws] * np.sort(PD3_lower)[::-1][:nws])
                    Gmin3 = np.sum(dd_flat_desc[:nws] * np.sort(PD3_lower)[:nws])
                    gc3.append((G3 - Gmin3) / (Gmax3 - Gmin3))
                if do_gc4:
                    G4 = np.sum(PD4 * dd)
                    PD4_lower = PD4[np.tril_indices(n, k=-1)]
                    Gmax4 = np.sum(dd_flat_desc[:nws] * np.sort(PD4_lower)[::-1][:nws])
                    Gmin4 = np.sum(dd_flat_desc[:nws] * np.sort(PD4_lower)[:nws])
                    gc4.append((G4 - Gmin4) / (Gmax4 - Gmin4))

    if do_wp:
        K = len(crr)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_forward  = (crr[1:K-1] - crr[0:K-2]) / (1 - crr[0:K-2])
            ratio_backward = (crr[2:K]   - crr[1:K-1]) / (1 - crr[1:K-1])
            WPI   = ratio_forward / np.maximum(0, ratio_backward)
            WPCI2 = ratio_forward - ratio_backward

        WPI[np.isnan(WPI)] = np.inf
        WPCI2[np.isnan(WPCI2)] = 0.0

        WPCI3 = WPI.copy()
        if np.sum(np.isfinite(WPI)) == 0:
            WPCI3[WPI == np.inf] = WPCI2[WPI == np.inf]
            WPCI3[WPI == -np.inf] = np.minimum(0, WPCI2[WPI == -np.inf])
        else:
            if np.max(WPI) < np.inf:
                if np.min(WPI) == -np.inf:
                    WPCI3[WPI == -np.inf] = np.min(WPI[np.isfinite(WPI)])
            if np.max(WPI) == np.inf:
                WPCI3[WPI == np.inf] = np.max(WPI[np.isfinite(WPI)]) + WPCI2[WPI == np.inf]
                WPCI3[WPI < np.inf] = WPI[WPI < np.inf] + WPCI2[WPI < np.inf]
                if np.min(WPI) == -np.inf:
                    WPCI3[WPI == -np.inf] = np.min(WPI[np.isfinite(WPI)]) + WPCI2[WPI == -np.inf]

    c_range = list(range(cmin, cmax + 1))

    WPC_df = pd.DataFrame({"c": range(cmin - 1, cmax + 2), "WPC": crr}) if do_wp else pd.DataFrame()
    WP_df = pd.DataFrame({"c": c_range,"WP": WPCI3}) if do_wp else pd.DataFrame()
    WPCI1_df = pd.DataFrame({"c": c_range,"WPI1": WPI}) if do_wp else pd.DataFrame()
    WPCI2_df = pd.DataFrame({"c": c_range, "WPCI2": WPCI2}) if do_wp else pd.DataFrame()

    XB_df = pd.DataFrame({"c": c_range, "XB": xb})if do_xb else pd.DataFrame()
    KWON_df = pd.DataFrame({"c": c_range, "KWON":kwon})  if do_kwon  else pd.DataFrame()
    KWON2_df = pd.DataFrame({"c": c_range, "KWON2": kwon2}) if do_kwon2 else pd.DataFrame()
    TANG_df  = pd.DataFrame({"c": c_range, "TANG":  tang})  if do_tang  else pd.DataFrame()
    HF_df    = pd.DataFrame({"c": c_range, "HF": hf}) if do_hf else pd.DataFrame()
    WL_df    = pd.DataFrame({"c": c_range, "WL": wl}) if do_wl else pd.DataFrame()
    PBM_df   = pd.DataFrame({"c": c_range, "PBM": pbm}) if do_pbm else pd.DataFrame()
    KPBM_df  = pd.DataFrame({"c": c_range, "KPBM": kpbm}) if do_kpbm else pd.DataFrame()

    CCVP_df  = pd.DataFrame({"c": c_range, "CCVP":ccvp}) if ("all" in indexlist or "CCVP" in indexlist) else pd.DataFrame()
    CCVS_df  = pd.DataFrame({"c": c_range, "CCVS":ccvs}) if ("all" in indexlist or "CCVS" in indexlist) else pd.DataFrame()

    GC1_df   = pd.DataFrame({"c": c_range, "GC1": gc1}) if do_gc1 else pd.DataFrame()
    GC2_df   = pd.DataFrame({"c": c_range, "GC2": gc2}) if do_gc2 else pd.DataFrame()
    GC3_df   = pd.DataFrame({"c": c_range, "GC3": gc3}) if do_gc3 else pd.DataFrame()
    GC4_df   = pd.DataFrame({"c": c_range, "GC4": gc4}) if do_gc4 else pd.DataFrame()

    IDX_list = {
        "WPC": WPC_df, "WP": WP_df, "WPCI1": WPCI1_df, "WPCI2": WPCI2_df,
        "XB": XB_df, "KWON": KWON_df, "KWON2": KWON2_df, "TANG": TANG_df,
        "HF": HF_df, "WL": WL_df, "PBM": PBM_df, "KPBM": KPBM_df,
        "CCVP": CCVP_df, "CCVS": CCVS_df,
        "GC1": GC1_df, "GC2": GC2_df, "GC3": GC3_df, "GC4": GC4_df
    }

    if "all" in indexlist:
        return IDX_list
    return {k: IDX_list[k] for k in indexlist}