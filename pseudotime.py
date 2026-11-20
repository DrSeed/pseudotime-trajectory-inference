# Compute pseudotime from an expression matrix using a kNN geodesic ordering.
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path


def compute_pseudotime(expr, n_components=10, n_neighbors=15, root_index=0):
    # expr: cells x genes numpy array. Returns pseudotime per cell and the PCA embedding.
    x = StandardScaler().fit_transform(expr)
    n_comp = min(n_components, x.shape[1], x.shape[0] - 1)
    emb = PCA(n_components=n_comp, random_state=0).fit_transform(x)
    graph = kneighbors_graph(emb, n_neighbors=n_neighbors, mode='distance', include_self=False)
    dist = shortest_path(graph, method='D', directed=False, indices=root_index)
    finite = np.isfinite(dist)
    if not finite.all():
        # Disconnected cells get the max finite distance so they sort last.
        dist[~finite] = dist[finite].max() if finite.any() else 0.0
    pt = dist / dist.max() if dist.max() > 0 else dist
    return pt, emb


def pick_root(expr, marker_index):
    # Choose the cell with the highest value of a naive marker gene as the root.
    return int(np.argmax(expr[:, marker_index]))


def main():
    parser = argparse.ArgumentParser(description='Compute pseudotime from an expression CSV (rows=cells, cols=genes).')
    parser.add_argument('--input', required=True, help='CSV with cells as rows and genes as columns.')
    parser.add_argument('--output', default='pseudotime.csv', help='Where to write per-cell pseudotime.')
    parser.add_argument('--neighbors', type=int, default=15)
    parser.add_argument('--components', type=int, default=10)
    parser.add_argument('--root', type=int, default=0, help='Row index of the root cell.')
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    pt, _ = compute_pseudotime(df.values, n_components=args.components,
                               n_neighbors=args.neighbors, root_index=args.root)
    out = pd.DataFrame({'cell': df.index, 'pseudotime': pt})
    out.to_csv(args.output, index=False)
    print('Wrote pseudotime for ' + str(len(out)) + ' cells to ' + args.output)


if __name__ == '__main__':
    main()
