import itertools
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

from pyspa.fembv import FEMBVKMeans

N_SAMPLES = 6000
N_SWITCHES = 2

N_INIT = 100

N_FEATURES = 2
N_CLUSTERS = 3
CLUSTER_1_MEAN = np.array([0, 0.5])
CLUSTER_1_COVARIANCE = np.array([[0.001, 0], [0, 0.02]])
CLUSTER_2_MEAN = np.array([0, -0.5])
CLUSTER_2_COVARIANCE = np.array([[0.001, 0], [0, 0.02]])
CLUSTER_3_MEAN = np.array([0.25, 0])
CLUSTER_3_COVARIANCE = np.array([[0.002, 0], [0, 0.3]])

CLUSTER_MEANS = np.vstack([CLUSTER_1_MEAN, CLUSTER_2_MEAN, CLUSTER_3_MEAN])
CLUSTER_COVARIANCES = np.stack(
    [CLUSTER_1_COVARIANCE, CLUSTER_2_COVARIANCE, CLUSTER_3_COVARIANCE], axis=0)


def generate_data(n_samples, n_switches=2, random_state=None):
    rng = check_random_state(random_state)
    run_length = int(np.floor(n_samples / (n_switches + 1)))

    Gamma = np.zeros((n_samples, N_CLUSTERS))
    cluster = 0
    for i in range(n_switches):
        Gamma[i * run_length:(i + 1) * run_length, cluster] = 1
        cluster = (cluster + 1) % N_CLUSTERS
    Gamma[(n_switches) * run_length:, cluster] = 1

    X = np.zeros((n_samples, N_FEATURES))
    for i in range(n_samples):
        r = np.zeros((N_CLUSTERS, N_FEATURES))
        for j in range(N_CLUSTERS):
            r[j] = rng.multivariate_normal(CLUSTER_MEANS[j],
                                           CLUSTER_COVARIANCES[j])
        X[i, :] = np.dot(Gamma[i, :], r)

    return X, Gamma


def run_kmeans(X, n_clusters=2, n_init=10, random_state=None):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                    random_state=random_state).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_


def run_fembv_kmeans(X, n_clusters=2, max_tv_norm=2, n_init=10,
                     random_state=None):
    best_cost = None
    best_model = None
    best_Gamma = None
    for i in range(n_init):
        model = FEMBVKMeans(n_components=n_clusters, max_tv_norm=max_tv_norm,
                            random_state=random_state, fem_basis='constant', verbose=1)
        Gamma = model.fit_transform(X)
        cost = model.cost_
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_model = model
            best_Gamma = Gamma
    return best_model.components_, best_Gamma


def plot_results(X, Gamma_true, kmeans_centroids, kmeans_labels, fembv_centroids,
    fembv_affs):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8), squeeze=False)
    fig.subplots_adjust(hspace=0.3)

    n_samples = X.shape[0]
    true_affs = np.zeros(n_samples, dtype='i8')

    markers = itertools.cycle(('.', '+', 's'))

    for i in range(N_CLUSTERS):
        true_affs[Gamma_true[:, i] == 1] = i + 1
        cluster_data = X[Gamma_true[:, i] == 1]
        ax[0, 0].plot(cluster_data[:, 0], cluster_data[:, 1],
                      marker=next(markers), linestyle='none',
                      label='Cluster {:d}'.format(i + 1))

    ax[0, 0].set_xlabel(r'$x_1$')
    ax[0, 0].set_ylabel(r'$x_2$')
    ax[0, 0].legend(numpoints=1)
    ax[0, 0].set_title(r'Synthetic data')

    ax[0, 1].plot(true_affs)
    ax[0, 1].yaxis.set_major_locator(MultipleLocator(1))
    ax[0, 1].set_xlabel(r'Time')
    ax[0, 1].set_ylabel(r'Cluster affiliation')
    ax[0, 1].set_title(r'True affiliations')

    colors = itertools.cycle(('r', 'g', 'b', 'y', 'c', 'k'))
    kmeans_n_clusters = kmeans_centroids.shape[0]
    for j in range(kmeans_n_clusters):
        c = next(colors)
        mask = kmeans_labels == j
        for i in range(N_CLUSTERS):
            cluster_data = X[np.logical_and(true_affs == i + 1, mask)]
            ax[1, 0].plot(cluster_data[:, 0], cluster_data[:, 1],
                          marker=next(markers), color=c, linestyle='none')
        ax[1, 0].plot(kmeans_centroids[j, 0], kmeans_centroids[j, 1], 'kx')

    ax[1, 0].set_xlabel(r'$x_1$')
    ax[1, 0].set_ylabel(r'$x_2$')
    ax[1, 0].set_title(r'k-means clusters')

    colors = itertools.cycle(('r', 'g', 'b', 'y', 'c', 'k'))
    fembv_n_clusters = fembv_centroids.shape[0]
    vp = np.argmax(fembv_affs, axis=1)
    for j in range(fembv_n_clusters):
        c = next(colors)
        mask = vp == j
        for i in range(N_CLUSTERS):
            cluster_data = X[np.logical_and(true_affs == i + 1, mask)]
            ax[1, 1].plot(cluster_data[:, 0], cluster_data[:, 1],
                          marker=next(markers), color=c, linestyle='none')
        ax[1, 1].plot(fembv_centroids[j, 0], fembv_centroids[j, 1], 'kx')

    ax[1, 1].set_xlabel(r'$x_1$')
    ax[1, 1].set_ylabel(r'$x_2$')
    ax[1, 1].set_title(r'FEM-BV-k-means clusters')

    plt.show()
    plt.close()


def main():
    random_state = 0
    n_samples = N_SAMPLES
    n_switches = N_SWITCHES
    X, Gamma_true = generate_data(n_samples, n_switches=n_switches,
                                  random_state=random_state)

    kmeans_centroids, kmeans_labels = run_kmeans(
        X, n_clusters=N_CLUSTERS, n_init=N_INIT)
    fembv_centroids, fembv_affs = run_fembv_kmeans(
        X, n_clusters=N_CLUSTERS, max_tv_norm=N_SWITCHES, n_init=1)

    plot_results(X, Gamma_true, kmeans_centroids, kmeans_labels,
        fembv_centroids, fembv_affs)


if __name__ == '__main__':
    main()
