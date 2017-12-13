import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
b_dir = '../epicurious-recipes-with-rating-and-nutrition/'

recipes = pd.read_csv(b_dir + 'epi_r.csv')
recipes.dropna(axis=0, inplace=True)


desc = recipes.describe().T

cat_cols = desc[desc['max'] == 1]

N = 50
cols_N = [col for col, _ in cat_cols.loc[:, ['mean']].sort_values(
    'mean', ascending=False)[:N].iterrows()]


from pystruct.learners import OneSlackSSVM
from pystruct.models import MultiLabelClf
from pystruct.datasets import load_scene
from sklearn.metrics import mutual_info_score
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import sparse
import networkx as nx

# https://pystruct.github.io/auto_examples/multi_label.html#sphx-glr-auto-examples-multi-label-py


def chow_liu_tree(y_):
    # compute mutual information using sklearn
    n_labels = y_.shape[1]
    mi = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    edges = np.vstack(mst.nonzero()).T
    edges.sort(axis=1)
    return edges

# https://github.com/JohnReid/pybool/blob/7ee0ec1b669ec0259405d3c120ec3fc2827ba397/python/pybool/chow_liu_trees.py


def build_chow_liu_tree(df, abs_weight=True):
    """
    Build a Chow-Liu tree from the data, X. n is the number of features. The weight on each edge is
    the negative of the mutual information between those features. The tree is returned as a networkx
    object.
    """
    G = nx.Graph()
    for u in df.columns:
        G.add_node(u)
        for v in df.columns:
            G.add_edge(u, v, weight=-mutual_info_score(df[u], df[v]))
    T = nx.minimum_spanning_tree(G)
    if abs_weight:
        for u, v, d in T.edges(data=True):
            T[u][v]['weight'] = abs(d['weight'])
    return T


# df.values
edges = chow_liu_tree(np.array(recipes[cols_N]))
G = build_chow_liu_tree(recipes[cols_N])

from tastu_teche.plt_show import plt_G, plt_show, plt_figure
plt_figure(15)
plt_G(G, pos_scale=2, width_scale=50, edge_alpha=0.5)
plt_show('chow_liu_tree.png')
