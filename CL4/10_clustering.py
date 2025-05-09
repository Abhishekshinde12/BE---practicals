from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()

# Features
X = iris.data

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit KMeans to the data
kmeans.fit(X)

# Get the cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualizing the clusters (asssuming 2D data for simplicity)
# plotting sepal length and sepal width
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
# printing centroid based only first 2 features
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
# Adding title, x label and y label
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
