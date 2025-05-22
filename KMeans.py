#KMEANS
import pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Create and save simple data
X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
pd.DataFrame(X, columns=['x', 'y']).to_csv('data.csv', index=False)

# Load CSV
data = pd.read_csv('data.csv')

# Apply KMeans
k = 4
model = KMeans(n_clusters=k, random_state=0)
labels = model.fit_predict(data)
score = silhouette_score(data, labels)
samples = silhouette_samples(data, labels)

# Plot
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5))

# Silhouette
a.set_title('Silhouette')
a.set_xlabel('Score')
a.set_yticks([])
y = 10
colors = ['yellow', 'green', 'blue', 'black']
for i in range(k):
    val = samples[labels == i]
    a.barh(range(y, y + len(val)), sorted(val), color=colors[i])
    y += len(val) + 10
a.axvline(score, color='red', linestyle='--')

# Clusters
b.set_title('Clusters')
b.scatter(data.x, data.y, c=labels, cmap='viridis')
centers = model.cluster_centers_
b.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X')
plt.tight_layout(); plt.show()
