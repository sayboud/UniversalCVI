# UniversalCVI

Algorithms for checking the accuracy of a clustering result with known classes, computing cluster validity indices, and generating plots for comparing them.

## Description

**UniversalCVI** package is an effective tool for evaluating clustering results by several cluster validity indices. It has functions to compute several indices as listed below for a user specified range of numbers of clusters and compare them in grid plots. The package is compatible with six clustering methods including K-means, fuzzy C-means, EM clustering, and hierarchical clustering (single, average, and complete linkage). Moreover, the UniversalCVI package has a function that computes the accuracy of clustering results when the true classes are known.

UniversalCVI requires Python 3.8 or higher and depends on NumPy, Pandas, SciPy, scikit-learn, and Matplotlib. It implements Fuzzy C-Means as a scikit-learn compatible estimator and provides a Gaussian Mixture Model wrapper with hierarchical initialization and BIC-based covariance model selection inspired by [`mclust`](https://CRAN.R-project.org/package=mclust).

In addition to the evaluation tools, the UniversalCVI package also includes 17 simulated datasets intially used for testing and comparing cluster validity indices in several perspectives written in Wiroonsri(2024) and Wiroonsri and Preedasawakul(2023). 

The cluster validity indices available in this package are listed as follows:

**Hard clustering:**

Dunn's index, Calinski–Harabasz index, Davies–Bouldin’s index, Point biserial correlation index, Chou-Su-Lai measure, Davies–Bouldin*’s index, Score function, Starczewski index, Pakhira–Bandyopadhyay–Maulik (for crisp clustering) index, Silhouette index, and Wiroonsri index.

**Fuzzy clustering:**

Xie–Beni index, KWON index, KWON2 index, TANG index , HF index, Wu–Li index, Pakhira–Bandyopadhyay–Maulik (for fuzzy clustering) index, KPBM index, Correlation Cluster Validity index, Generalized C index, Wiroonsri and Preedasawakul index.


## References

N. Wiroonsri and O. Preedasawakul (2026). UniversalCVI: Hard and Soft Cluster Validity Indices. R package version 1.4.0. https://CRAN.R-project.org/package=UniversalCVI

N. Wiroonsri and O. Preedasawakul (2026). A correlation-based fuzzy cluster validity index with secondary options detector. Fuzzy Sets and Systems, 523, 109632. https://doi.org/10.1016/j.fss.2025.109632

Nathakhun Wiroonsri. (2024). Clustering performance analysis using a new correlation-based cluster validity index. Pattern Recognition, 145, 109910. https://doi.org/10.1016/j.patcog.2023.109910

## License

The UniversalCVI package as a whole is distributed under [GPL(>=3)](https://www.gnu.org/licenses/gpl-3.0.en.html).
