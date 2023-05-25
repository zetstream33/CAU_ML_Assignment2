import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets, metrics
from sklearn.manifold import TSNE
import seaborn as sns


fashion_mnist = datasets.fetch_openml('Fashion-MNIST')
X_test = fashion_mnist.data[60000:]
y_test = fashion_mnist.target[60000:]


def run_and_plot_pca_kmeans(X, y, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=10, random_state=42)
    y_pred = kmeans.fit_predict(X_pca)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_pred, palette=sns.color_palette("hls", 10))
    plt.title(f't-SNE visualization of K-means clustering with PCA={n_components}')
    plt.show()

    ari_score = adjusted_rand_score(y, y_pred)
    print(f'Adjusted Rand Index (ARI) with PCA={n_components}: {ari_score}')

    return ari_score

dimensions = [784, 100, 50, 10]
ari_scores = {}

for dim in dimensions:
    ari_scores[dim] = run_and_plot_pca_kmeans(X_test, y_test, dim)

ari_df = pd.DataFrame(list(ari_scores.items()), columns=['PCA_Dimension', 'ARI_Score'])
print(ari_df)
