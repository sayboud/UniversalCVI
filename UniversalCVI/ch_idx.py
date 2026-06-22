import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage as scipy_linkage, cut_tree
from ._utils import valid_methods, valid_linkages

def CH_idx(x, kmax, kmin = 2, method = "kmeans", linkage = None, n_init = 100):
    x = np.array(x)
    dm = x.shape

    if not isinstance(kmax,int):
        raise TypeError("Argument 'kmax' must be an integer")
    if kmax > dm[0]:
        raise ValueError("The maximum number of clusters for consideration should be less than or equal to the number of data points in dataset.")
    if not isinstance(kmin,int):
        raise TypeError("Argument 'kmin' must be an integer")
    if kmin <= 1:
        warnings.warn("The minimum number of clusters for consideration should be more than 1")
    if method not in valid_methods:
        raise ValueError(f"Argument 'method' should be one of {valid_methods}")
    if method == "kmeans":
        if not isinstance(n_init,int):
            raise TypeError("Argument 'n_init' must be an integer")
        if linkage is not None:
            raise ValueError("Argument 'linkage' must be None when using K-means")
        
    if method == "hclust":
        if linkage not in valid_linkages:
            raise ValueError(f"Argument 'linkage' should be one of {valid_linkages}")
        
        model = scipy_linkage(x, method=linkage)
        
    ch = []

    global_mean = np.mean(x,axis=0)

    for k in range(kmin, kmax + 1):
        xnew = np.zeros((dm[0], dm[1]))
        centroid = np.zeros((k, dm[1]))

        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=n_init)
            model.fit(x)
            cluss = model.labels_
            centroid = model.cluster_centers_
            xnew = centroid[cluss]

        elif method == "hclust":
            cluss = cut_tree(model, n_clusters=k).flatten()
            for j in range(k):
                pts = x[cluss == j]
                if pts.shape[0] == 1:
                    centroid[j] = pts[0]
                else:
                    centroid[j] = pts.mean(axis=0)
            xnew = centroid[cluss]

        if len(np.unique(cluss)) < k:
            warnings.warn("Some clusters are empty.")

        d_cen = np.sum((x - xnew)**2, axis=1)

        num = np.array([
            np.sum(cluss == i) * np.sum((centroid[i] - global_mean) ** 2)
            for i in range(k)
        ])
        
        dem = np.array([
            np.sum(d_cen[cluss == i])
            for i in range(k)
        ])
        
        num = num[~np.isnan(num)]
        
        ch.append(((dm[0] - k) / (k - 1)) * (np.sum(num) / np.sum(dem)))
    
    CH_data = pd.DataFrame({
        "k": range(int(kmin), int(kmax) + 1),
        "CH": ch})
    
    return CH_data