import matplotlib.pyplot as plt
import math
import numpy as np
from ._utils import valid_indexlist

max_indices = {"PBM", "KPBM", "CH", "DI", "PB", "NCI", "NCI1", "NCI2",
               "STR", "SH", "WP", "WPCI1", "WPCI2", "CCVP", "CCVS",
               "GC1", "GC2", "GC3", "GC4"}
no_marker_indices = {"NC","WPC"}
exclude = {"NC", "NCI1", "NCI2", "WPC", "WPCI1", "WPCI2"}

def plot_idx(idxresult, selected_idx=None):

    if not isinstance(idxresult, dict):
        raise TypeError("Argument 'idxresult' must be a dictionary of DataFrames")
    if not any(k in valid_indexlist for k in idxresult.keys()):
        raise ValueError("Bad input data, 'idxresult' is not a result from our package function.")

    if selected_idx is not None:
        if max(selected_idx) > len(idxresult):
            raise ValueError("selected_idx values must be less than or equal to the number of indices in idxresult")
        keys = list(idxresult.keys())
        idxresult = {keys[i-1]: idxresult[keys[i-1]] for i in selected_idx}

    n_total = len(idxresult)
    if n_total in [13, 18]:
        name_idx = [k for k in idxresult.keys() if k not in exclude and idxresult[k] is not None][:8]
    else:
        name_idx = [k for k in idxresult.keys() if idxresult[k] is not None][:8]

    n_idx = len(name_idx)

    if n_idx <= 3:
        nrows, ncols = 1, n_idx
    else:
        ncols = math.ceil(n_idx / 2)
        nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, idx_name in enumerate(name_idx):
        df = idxresult[idx_name]
        ax = axes[i]
        k_col = df.columns[0]
        v_col = df.columns[1]

        ax.plot(df[k_col], df[v_col], marker='o', color='black')
        ax.set_xlabel("c" if k_col == "c" else "k")
        ax.set_ylabel(idx_name)

        if idx_name in no_marker_indices:
            pass
        elif idx_name in max_indices:
            best_k = df.loc[df[v_col].idxmax(), k_col]
            best_v = df[v_col].max()
            ax.plot(best_k, best_v, 'ro', markersize=8)
        else:
            best_k = df.loc[df[v_col].idxmin(), k_col]
            best_v = df[v_col].min()
            ax.plot(best_k, best_v, 'ro', markersize=8)

    for j in range(n_idx, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()