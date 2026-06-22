import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from ._utils import valid_methods, valid_linkages, valid_corr, compute_corr, get_centroid, valid_indexlist

def Hvalid(x, kmax, kmin=2, indexlist="all", method="kmeans", p=2, q=2,
           corr="pearson", n_init=100, sampling=1, NCstart=True, linkage=None):
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
    if not any(idx in valid_indexlist for idx in ([indexlist] if isinstance(indexlist, str) else indexlist)):
        raise ValueError("Argument 'indexlist' is not in the possible value")
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

    if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
           for idx in ["all", "DB", "DBs"]):
        if not isinstance(p, (int, float)):
            raise TypeError("Argument 'p' must be numeric")
        if not isinstance(q, (int, float)):
            raise TypeError("Argument 'q' must be numeric")

    if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
           for idx in ["all", "NC", "NCI", "NCI1", "NCI2", "PB", "STR"]):
        if corr not in valid_corr:
            raise ValueError(f"Argument 'corr' should be one of {valid_corr}")
        if not isinstance(NCstart, bool):
            raise TypeError("Argument 'NCstart' must be a boolean")
        if not isinstance(sampling, (int, float)):
            raise TypeError("Argument 'sampling' must be numeric")
        if not (0 < sampling <= 1):
            raise ValueError("'sampling' must be greater than 0 and less than or equal to 1")
        if sampling < 1:
            n_sample = int(np.ceil(dm[0] * sampling))
            sample_idx = np.random.choice(dm[0], n_sample, replace=False)
            x = x[sample_idx]
        d = pdist(x)
        dtom = np.sqrt(np.sum((x - x.mean(axis=0)) ** 2, axis=1))
        E0 = np.sum(dtom)

    pb   = []
    ch   = []
    db   = []
    dbs  = []
    sf   = []
    di   = []
    sc   = []
    crr  = np.zeros(kmax - kmin + 3)
    str_vals = np.zeros(kmax - kmin + 1)
    pbm  = np.zeros(kmax - kmin + 1)
    EK   = np.zeros(kmax - kmin + 3)
    DK   = np.zeros(kmax - kmin + 4)
    md   = np.zeros(kmax - kmin + 4)
    csl  = np.zeros(kmax - kmin + 1)

    if method == "hclust":
        H_model = scipy_linkage(x, method=linkage)

    needs_nc  = any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
                    for idx in ["all", "NC", "NCI", "NCI1", "NCI2"])
    needs_str = any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
                    for idx in ["all", "STR", "PBM"])

    if needs_nc or needs_str:
        if kmin <= 2:
            if needs_nc:
                if NCstart:
                    crr[0] = np.std(dtom, ddof=1) / (np.max(dtom) - np.min(dtom))
                else:
                    crr[0] = 0
            if needs_str:
                EK[0] = E0
        else:
            if method == "kmeans":
                model = KMeans(n_clusters=kmin-1, n_init=n_init)
                model.fit(x)
                cluss = model.labels_
                centroid = model.cluster_centers_
                xnew = centroid[cluss]
            elif method == "hclust":
                cluss = cut_tree(H_model, n_clusters=kmin-1).flatten()
                centroid = get_centroid(x, cluss, kmin-1,dm[1])
                xnew = centroid[cluss]
            if needs_nc:
                crr[0] = compute_corr(d, pdist(xnew), corr)
            if needs_str:
                EK[0] = np.sum(np.sqrt(np.sum((x - xnew) ** 2, axis=1)))

        if method == "kmeans":
            model = KMeans(n_clusters=kmax+1, n_init=n_init)
            model.fit(x)
            cluss = model.labels_
            centroid = model.cluster_centers_
            xnew = centroid[cluss]
        elif method == "hclust":
            cluss = cut_tree(H_model, n_clusters=kmax+1).flatten()
            centroid = get_centroid(x, cluss, kmax+1,dm[1])
            xnew = centroid[cluss]
        if needs_nc:
            crr[kmax - kmin + 2] = compute_corr(d, pdist(xnew), corr)
        if needs_str:
            ddd = pdist(centroid)
            md[kmax - kmin + 2] = np.max(ddd)
            DK[kmax - kmin + 2] = np.max(ddd) / np.min(ddd)

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
            centroid = get_centroid(x, cluss, k,dm[1])
            xnew = centroid[cluss]

        if len(np.unique(cluss)) < k:
            warnings.warn("Some clusters are empty.")

        if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
               for idx in ["all", "CSL"]):
            for j in range(k):
                pts = x[cluss == j]
                dd = squareform(pdist(pts))
                csl[k - kmin] += np.sum(np.max(dd, axis=1)) / np.sum(cluss == j)
            dd2 = squareform(pdist(centroid))
            np.fill_diagonal(dd2, np.inf)
            csl[k - kmin] /= np.sum(np.min(dd2, axis=0))

        if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
               for idx in ["all", "CH"]):
            global_mean = x.mean(axis=0)
            d_cen = np.sum((x - xnew) ** 2, axis=1)
            num = np.array([np.sum(cluss == i) * np.sum((centroid[i] - global_mean) ** 2)
                            for i in range(k)])
            dem = np.array([np.sum(d_cen[cluss == i]) for i in range(k)])
            num = num[~np.isnan(num)]
            ch.append(((dm[0] - k) / (k - 1)) * (np.sum(num) / np.sum(dem)))

        if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
               for idx in ["all", "SH"]):
            size = np.bincount(cluss)
            sss = np.zeros(dm[0])
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

        if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
               for idx in ["all", "DB", "DBs", "SF"]):
            sizecluss = np.bincount(cluss)
            S = np.zeros(k)
            for i in range(k):
                C = sizecluss[i]
                if C > 1:
                    cenI = xnew[cluss == i]
                    S[i] = (np.sum(np.sqrt(np.sum((x[cluss == i] - cenI) ** 2, axis=1)) ** q) / C) ** (1/q)
                else:
                    S[i] = 0

            m = squareform(pdist(centroid, metric="minkowski", p=p))
            r = np.zeros(k)
            rs = np.zeros(k)
            wcdd = np.zeros(k)
            for i in range(k):
                mask = np.arange(k) != i
                m_row = m[i, mask]
                r[i] = np.max((S[i] + S[mask]) / m_row)
                rs[i] = np.max(S[i] + S[mask]) / np.min(m_row)
                pts = x[cluss == i]
                stacked = np.vstack([centroid[i], pts])
                wcdd[i] = np.sum(pdist(stacked)[:sizecluss[i]]) / sizecluss[i]

            db.append(np.mean(r))
            dbs.append(np.mean(rs))
            stacked_global = np.vstack([x.mean(axis=0), centroid])
            bcd = np.sum(pdist(stacked_global)[:k] * sizecluss) / (dm[0] * k)
            wcd = np.sum(wcdd)
            sf.append(1 - 1 / np.exp(bcd + wcd))

        if needs_str:
            EK[k - kmin + 1] = np.sum(np.sqrt(np.sum((x - xnew) ** 2, axis=1)))
            ddd = pdist(centroid)
            md[k - kmin + 1] = np.max(ddd)
            DK[k - kmin + 1] = np.max(ddd) / np.min(ddd)

        if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
               for idx in ["all", "DI"]):
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

        if any(idx in ([indexlist] if isinstance(indexlist, str) else indexlist)
               for idx in ["all", "PB"]):
            d3 = pdist(xnew)
            d3[d3 > 0] = 1
            pb.append(compute_corr(d, d3, corr))

        if needs_nc:
            d2 = pdist(xnew)
            crr[k - kmin + 1] = compute_corr(d, d2, corr)

    if needs_str:
        EKK = E0 / EK
        str_vals = (EKK[1:len(EKK)-1] - EKK[0:len(EKK)-2]) * \
                   (DK[2:len(DK)-1] - DK[1:len(DK)-2])
        pbm = EKK[1:len(EKK)-1] * md[1:len(EKK)-1] / np.arange(kmin, kmax + 1)

    if needs_nc:
        K = len(crr)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_forward = (crr[1:K-1] - crr[0:K-2]) / (1 - crr[0:K-2])
            ratio_backward = (crr[2:K] - crr[1:K-1]) / (1 - crr[1:K-1])
            NWI = ratio_forward / np.maximum(0, ratio_backward)
        NWI2 = ratio_forward - ratio_backward
        NWI3 = NWI.copy()
        if np.max(NWI) < np.inf:
            if np.min(NWI) == -np.inf:
                NWI3[NWI == -np.inf] = np.min(NWI[np.isfinite(NWI)])
        if np.max(NWI) == np.inf:
            NWI3[NWI == np.inf] = np.max(NWI[np.isfinite(NWI)]) + NWI2[NWI == np.inf]
            NWI3[NWI < np.inf] = NWI[NWI < np.inf] + NWI2[NWI < np.inf]
            if np.min(NWI) == -np.inf:
                NWI3[NWI == -np.inf] = np.min(NWI[np.isfinite(NWI)]) + NWI2[NWI == -np.inf]
        NC  = pd.DataFrame({"k": range(kmin-1, kmax+2), "NC": crr})
        NCI1 = pd.DataFrame({"k": range(kmin, kmax+1), "NCI1": NWI})
        NCI2 = pd.DataFrame({"k": range(kmin, kmax+1), "NCI2": NWI2})
        NCI  = pd.DataFrame({"k": range(kmin, kmax+1), "NCI": NWI3})
    else:
        NC = NCI = NCI1 = NCI2 = 0

    k_range = range(kmin, kmax + 1)
    my_list = {
        "NC":   NC,
        "NCI":  NCI,
        "NCI1": NCI1,
        "NCI2": NCI2,
        "CSL":  pd.DataFrame({"k": k_range, "CSL":  csl}) if any(i in indexlist for i in ["all", "CSL"]) else None,
        "STR":  pd.DataFrame({"k": k_range, "STR":  str_vals}) if needs_str else None,
        "CH":   pd.DataFrame({"k": k_range, "CH":   ch}) if any(i in indexlist for i in ["all", "CH"]) else None,
        "SH":   pd.DataFrame({"k": k_range, "SH":   sc}) if any(i in indexlist for i in ["all", "SH"]) else None,
        "DB":   pd.DataFrame({"k": k_range, "DB":   db}) if any(i in indexlist for i in ["all", "DB", "DBs"]) else None,
        "DBs":  pd.DataFrame({"k": k_range, "DBs":  dbs}) if any(i in indexlist for i in ["all", "DB", "DBs"]) else None,
        "PBM":  pd.DataFrame({"k": k_range, "PBM":  pbm}) if needs_str else None,
        "DI":   pd.DataFrame({"k": k_range, "DI":   di}) if any(i in indexlist for i in ["all", "DI"]) else None,
        "PB":   pd.DataFrame({"k": k_range, "PB":   pb}) if any(i in indexlist for i in ["all", "PB"]) else None,
        "SF":   pd.DataFrame({"k": k_range, "SF":   sf}) if any(i in indexlist for i in ["all", "DB", "DBs", "SF"]) else None,
}

    if indexlist == "all":
        return my_list
    else:
        if isinstance(indexlist, str):
            indexlist = [indexlist]
        return {idx: my_list[idx] for idx in indexlist}