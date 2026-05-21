import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=150, centers=4, cluster_std=0.8, random_state=42)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = agg.fit_predict(X)

# Dendrogram (small subset for clarity)
Z = linkage(X[:50], method='ward')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
scatter = axes[0].scatter(X[:,0], X[:,1], c=labels, cmap='tab10', alpha=0.7)
axes[0].set_title('Hierarchical Clustering (Ward)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
plt.colorbar(scatter, ax=axes[0])

# Dendrogram
dendrogram(Z, ax=axes[1], truncate_mode='level', p=5,
           leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7*max(Z[:,2]))
axes[1].set_title('Dendrogram (Ward Linkage)')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Distance')

plt.tight_layout()
plt.show()

# Compare linkage methods
linkage_methods = ['ward', 'complete', 'average', 'single']
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, method in zip(axes, linkage_methods):
    cl = AgglomerativeClustering(n_clusters=4, linkage=method)
    lbl = cl.fit_predict(X)
    ax.scatter(X[:,0], X[:,1], c=lbl, cmap='tab10', alpha=0.7, s=15)
    ax.set_title(f'{method.capitalize()} linkage')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Linkage Method Comparison')
plt.tight_layout()
plt.show()
