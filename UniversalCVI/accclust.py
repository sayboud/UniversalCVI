import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from UniversalCVI.clustering import FCMeans, bic_gm

def AccClust(x, label_col="label", algorithm="Kmeans", fzm=2, scale=True, n_init=100, max_iter=100):

    if not isinstance(x, pd.DataFrame):
        raise TypeError("Argument 'x' must be a pandas DataFrame with a label column")
    if not isinstance(label_col, str):
        raise TypeError("Argument 'label_col' must be a string indicating the true label column name")
    if label_col not in x.columns:
        raise ValueError(f"Column '{label_col}' is not in the data frame")

    valid_algorithms = ["FCM", "EM", "Kmeans"]
    if isinstance(algorithm, str):
        algorithm = [algorithm]
    if not any(alg in valid_algorithms for alg in algorithm):
        raise ValueError("Argument 'algorithm' should be one of 'FCM', 'EM', 'Kmeans'")
    if not isinstance(scale, bool):
        raise TypeError("Argument 'scale' must be a boolean")
    if "FCM" in algorithm:
        if fzm <= 1:
            raise ValueError("Argument 'fzm' must be greater than 1")
        if not isinstance(n_init, int):
            raise TypeError("Argument 'n_init' must be an integer")
        if not isinstance(max_iter, int):
            raise TypeError("Argument 'max_iter' must be an integer")

    labels = pd.factorize(x[label_col])[0]
    k = len(np.unique(labels))
    u = pd.Series(labels).value_counts().sort_values(ascending=False).index.tolist()
    features = x.drop(columns=[label_col]).values
    if scale:
        features = (features - np.mean(features,axis=0)) / np.std(features,axis=0, ddof=1)

    result = []

    for alg in algorithm:
        if alg == "Kmeans":
            model = KMeans(n_clusters=k, n_init=n_init)
            model.fit(features)
            cluster = model.labels_

        elif alg == "EM":
            model = bic_gm(features, k=k)
            cluster = model.predict(features)

        elif alg == "FCM":
            model = FCMeans(n_clusters=k, n_init=n_init, m=fzm, max_iter=max_iter)
            model.fit(features)
            cluster = model.labels_

        cluster = cluster + 100
        cluster2 = cluster.copy()
        wold = []

        for i in range(k):
            mask = labels == u[i]
            ttt = pd.Series(cluster[mask]).value_counts()
            for w in ttt.index:
                if w not in wold:
                    break
            cluster2[cluster2 == w] = u[i]
            wold.append(w)

        acc = np.mean(cluster2 == labels)
        result.append({"ALGORITHM": alg, "ACC": acc})

    return pd.DataFrame(result)