import numpy as np
from scipy import stats

valid_methods = ["kmeans", "hclust"]
valid_fuzzy_methods = ["FCM", "EM"]
valid_linkages = ["ward", "average", "complete", "single"]
valid_corr = ["pearson", "kendall", "spearman"]
valid_indexlist = ["all", "WPC", "WP", "WPCI1", 
        "WPCI2", "XB", "KWON", "KWON2", "TANG", "HF", "WL", "PBM", 
        "KPBM", "CCVP", "CCVS", "GC1", "GC2", "GC3", "GC4", "NC", 
        "NCI", "NCI1", "NCI2", "CSL", "CH", "DB", "DBs", "DI", 
        "PB", "SF", "SH", "STR"]

def compute_corr(d1, d2, method="pearson"):
    if method == "pearson":
        return stats.pearsonr(d1, d2)[0]
    elif method == "kendall":
        return stats.kendalltau(d1, d2)[0]
    elif method == "spearman":
        return stats.spearmanr(d1, d2)[0]
    
def get_centroid(x, cluss, k, n_features):
    centroid = np.zeros((k, n_features))
    for j in range(k):
        pts = x[cluss == j]
        centroid[j] = pts[0] if pts.shape[0] == 1 else pts.mean(axis=0)
    return centroid