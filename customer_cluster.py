##moduler code

## importing the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## create the images folder if it does not exist
os.makedirs("images", exist_ok=True)

##generating the data
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.5, random_state=42)

##all k valus from 1 to 10

k_values = range(1, 11)
inertias = []
silhouettes = {}

for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes[k] = silhouette_score(X, km.labels_)


##plotting the inertia values

plt.figure(figsize=(7, 4.5))
plt.plot(list(k_values), inertias, marker="o", linewidth=2)
plt.xticks(list(k_values))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Curve")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("images/elbow_curve.png", dpi=350)
plt.show()

## selecting the best k
## K=4 chosen from the elbow curve (silhouette alone would suggest K=3 due to overlapping clusters)
best_k = 4

final_km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
cluster_labels = final_km.fit_predict(X)

plt.figure(figsize=(7, 5.5))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis", s=30)
plt.scatter(final_km.cluster_centers_[:, 0], final_km.cluster_centers_[:, 1], c="red", marker="x", s=300, linewidths=2, label="Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Customer Segments (K-Means, K={best_k})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("images/customer_clusters_best_k_4.png", dpi=300)
plt.show()