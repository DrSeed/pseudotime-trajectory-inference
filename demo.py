# Self-contained pseudotime demo: simulate branching cell states and infer an ordering.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path


def simulate_branching(n_per_arm=120, n_genes=40, seed=0):
    # Build one trunk and two branches sharing a common progenitor.
    rng = np.random.default_rng(seed)
    genes = n_genes

    def arm(latent, programs):
        # latent in [0,1]; programs is a list of (gene_idx, direction) tuples.
        base = rng.normal(0.0, 0.3, size=(len(latent), genes))
        for g, direction in programs:
            base[:, g] += direction * latent
        return base

    trunk_t = np.linspace(0.0, 1.0, n_per_arm)
    branch_t = np.linspace(0.0, 1.0, n_per_arm)

    # Shared early program on gene 0 (rises along trunk then stays high).
    trunk = arm(trunk_t, [(0, 3.0)])
    # Branch A activates gene 1, branch B activates gene 2.
    branch_a = arm(branch_t, [(0, 0.0), (1, 3.0)])
    branch_a[:, 0] += 3.0
    branch_b = arm(branch_t, [(0, 0.0), (2, 3.0)])
    branch_b[:, 0] += 3.0

    expr = np.vstack([trunk, branch_a, branch_b])
    labels = (['trunk'] * n_per_arm + ['branch_A'] * n_per_arm + ['branch_B'] * n_per_arm)
    # Keep gene 0 high-but-noisy so the trunk end is the clear naive root.
    return expr, np.array(labels)


def compute_pseudotime(expr, n_components=10, n_neighbors=15, root_index=0):
    x = StandardScaler().fit_transform(expr)
    n_comp = min(n_components, x.shape[1], x.shape[0] - 1)
    emb = PCA(n_components=n_comp, random_state=0).fit_transform(x)
    graph = kneighbors_graph(emb, n_neighbors=n_neighbors, mode='distance', include_self=False)
    dist = shortest_path(graph, method='D', directed=False, indices=root_index)
    finite = np.isfinite(dist)
    if not finite.all():
        dist[~finite] = dist[finite].max() if finite.any() else 0.0
    pt = dist / dist.max() if dist.max() > 0 else dist
    return pt, emb


def main():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    expr, labels = simulate_branching()
    # Root = cell with lowest early marker (gene 0): the most naive progenitor.
    root = int(np.argmin(expr[:, 0]))
    pt, emb = compute_pseudotime(expr, root_index=root)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=pt, cmap='viridis', s=18)
    axes[0].scatter(emb[root, 0], emb[root, 1], c='red', s=90, marker='*', label='root')
    axes[0].set_title('PCA embedding coloured by pseudotime')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(loc='best')
    fig.colorbar(sc, ax=axes[0], label='pseudotime')

    order = np.argsort(pt)
    window = 25
    for gene_idx, name, colour in [(0, 'early marker', 'black'),
                                   (1, 'branch A gene', 'tab:blue'),
                                   (2, 'branch B gene', 'tab:orange')]:
        vals = expr[order, gene_idx]
        smoothed = np.convolve(vals, np.ones(window) / window, mode='same')
        axes[1].plot(pt[order], smoothed, color=colour, label=name)
    axes[1].set_title('Gene expression along pseudotime')
    axes[1].set_xlabel('pseudotime')
    axes[1].set_ylabel('smoothed expression')
    axes[1].legend(loc='best')

    fig.tight_layout()
    fig.savefig('figures/demo.png', dpi=120)

    summary = pd.DataFrame({
        'branch': ['trunk', 'branch_A', 'branch_B'],
        'n_cells': [int((labels == b).sum()) for b in ['trunk', 'branch_A', 'branch_B']],
        'mean_pseudotime': [float(pt[labels == b].mean()) for b in ['trunk', 'branch_A', 'branch_B']],
    })
    summary.to_csv('results/summary.csv', index=False)
    print('Saved figures/demo.png and results/summary.csv')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
