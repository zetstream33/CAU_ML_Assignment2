import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets, metrics
from sklearn.manifold import TSNE
import seaborn as sns

fashion_mnist = datasets.fetch_openml('Fashion-MNIST')
X_test = fashion_mnist.data[60000:] / 255
y_test = fashion_mnist.target[60000:].astype(int)

def run_and_plot_pca_dbscan(X, y, n_components, eps=5, min_samples=4):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, p=5)
    y_pred = dbscan.fit_predict(X_pca)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_pred, palette=sns.color_palette("hls", 10))
    plt.title(f't-SNE visualization of DBSCAN clustering with PCA={n_components}')
    plt.show()

    ari_score = adjusted_rand_score(y, y_pred)
    print(f'Adjusted Rand Index (ARI) with PCA={n_components}: {ari_score}')
    print("y_qrep.unique(numpy) : ", np.unique(y_pred), ". If this value is -1, all points considered as noise!!!!")

    return ari_score

dimensions = [784, 100, 50, 10]
ari_scores = {}

for dim in dimensions:
    ari_scores[dim] = run_and_plot_pca_dbscan(X_test, y_test, dim)

ari_df = pd.DataFrame(list(ari_scores.items()), columns=['PCA_Dimension', 'ARI_Score'])
print(ari_df)
