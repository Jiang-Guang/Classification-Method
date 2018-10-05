import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture


def point_distance(p1:np.ndarray, p2:np.ndarray):
    return np.linalg.norm(p1 - p2)

def K_Means(X:np.ndarray, n_cluster=2):
    km = KMeans(n_clusters=n_cluster)
    km.fit_predict(X)
    label = km.labels_
    return label

def Mean_Shift(X:np.ndarray):
    ms = MeanShift()
    ms.fit_predict(X)
    label = ms.labels_
    return label

def Dbsacn(X:np.ndarray, min_simple=5):
    nbrs = NearestNeighbors(n_neighbors = min_simple + 1, algorithm='ball_tree', metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)
    eps = 2 * np.mean( distances[:,-1] )
    dbs = DBSCAN(eps=eps, min_samples=min_simple, metric="euclidean")
    dbs.fit_predict(X)
    label = dbs.labels_
    return label

def Gmm(X:np.ndarray):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X)
    label = gmm.predict(X)
    return label

def Agglomerative_Hierarchical_Clustering(X:np.ndarray):
    ahc = AgglomerativeClustering(n_clusters=2)
    ahc.fit_predict(X)
    label = ahc.labels_
    return label

def Spectral_Cluster(X:np.ndarray):
    sc = SpectralClustering(n_clusters=2)
    sc.fit_predict(X)
    label = sc.labels_
    return label


