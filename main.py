import method
import numpy as np
import matplotlib.pyplot as plt

def show_ret(X:np.ndarray, label:np.ndarray):
    plt.scatter(x=X[:,0], y=X[:,1], c=label)
    plt.show()

if __name__ == '__main__':
    data = np.load("./t3.npy")

    show_ret(data, label=np.zeros(len(data)))
    # test
    label = method.K_Means(data)
    show_ret(data, label)

    label = method.Mean_Shift(data)
    show_ret(data, label)

    label = method.Dbsacn(data)
    show_ret(data, label)

    label = method.Gmm(data)
    show_ret(data, label)

    label = method.Agglomerative_Hierarchical_Clustering(data)
    show_ret(data, label)


    label = method.Spectral_Cluster(data)
    show_ret(data, label)
