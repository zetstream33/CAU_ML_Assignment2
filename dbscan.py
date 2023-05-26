import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets, metrics
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import StandardScaler

fashion_mnist = datasets.fetch_openml('Fashion-MNIST')
X_test = fashion_mnist.data[60000:] / 255
y_test = fashion_mnist.target[60000:].astype(int)
#Normalize 하는 과정 거침


def run_and_plot_pca_dbscan(X, y, n_components, eps, min_samples):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Standardize the features to have mean=0 and variance=1
    scaler = StandardScaler()
    X_pca = scaler.fit_transform(X_pca)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
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
    # 전부 Noise나와서 Ari-score 0되는거 확인하는 코드

    return ari_score

dimensions = [784, 100, 50, 10]
eps = 27
min_samples = 3
ari_scores_dbscan = {}

for dim in dimensions:
    ari_scores_dbscan[dim] = run_and_plot_pca_dbscan(X_test, y_test, dim, eps, min_samples)


ari_df_dbscan = pd.DataFrame(list(ari_scores_dbscan.items()), columns=['PCA_Dimension', 'ARI_Score'])
print(ari_df_dbscan)

#    PCA_Dimension  ARI_Score   (eps= 27, min_samples = 3)
# 0            784   0.055443
# 1            100   0.000000
# 2             50   0.000000
# 3             10   0.000000



# def dbscan_revise(X, y, n_components, eps, min_samples):
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X)
#
#     model = TSNE(learning_rate=300)
#     transformed = model.fit_transform(X_pca)
#
#     model = DBSCAN(eps = eps, min_samples= min_samples)
#     predict = model.fit_predict(transformed)
#
#     tsne = TSNE(n_components=2, random_state=42)
#     X_tsne = tsne.fit_transform(X_pca)
#
#
#     dataset = pd.DataFrame({{'Column1:',transformed[:,0],'Column2:',transformed[:,1]})
#     dataset['cluster_num'] = pd.Series(predict.labels_)
#
#     viz_img(y_pred)
# def run_and_plot_pca_dbscan(X, y, n_components, eps, min_samples):
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X)
#
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     y_pred = dbscan.fit_predict(X_pca)
#
#     tsne = TSNE(n_components=2, random_state=42)
#     X_tsne = tsne.fit_transform(X_pca)
#
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_pred, palette=sns.color_palette("hls", 10))
#     plt.title(f't-SNE visualization of DBSCAN clustering with PCA={n_components}')
#     plt.show()
#
#     ari_score = adjusted_rand_score(y, y_pred)
#     print(f'Adjusted Rand Index (ARI) with PCA={n_components}: {ari_score}')
#     print("y_qrep.unique(numpy) : ", np.unique(y_pred), ". If this value is -1, all points considered as noise!!!!")
#     # 전부 Noise나와서 Ari-score 0되는거 확인하는 코드
#
#     return ari_score
#
# dimensions = [784, 100, 50, 10]
# ari_scores = {}
#
# for dim in dimensions:
#     ari_scores[dim] = dbscan_revise(X_test, y_test, dim, eps = 10, min_samples=4)
#
# ari_df = pd.DataFrame(list(ari_scores.items()), columns=['PCA_Dimension', 'ARI_Score'])
# print(ari_df)

