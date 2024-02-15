import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ================================================== #
#                 K-means algorithm                  #
# ================================================== #


# Randomly initializing K centroid by picking K samples from dataset
def initialize_random_centroids(K, dataset):
    m, n = np.shape(dataset)
    centroids = np.empty((K, n))
    for i in range(K):
        centroids[i] = dataset[np.random.choice(range(m))]
    return centroids


# Calculate the distance between two vectors (default euclidean distance)
def calculate_distance(x, y, P=np.eye(2)):
    diff = x - y
    P_inv = np.linalg.inv(np.dot(P.T, P))
    mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, P_inv), diff.T))

    return mahalanobis_dist


def closest_centroid(x, centroids, K, P):
    distances = np.empty(K)
    for i in range(K):
        distances[i] = calculate_distance(centroids[i], x, P)
    return np.argmin(distances)  # return the index of the lowest distance


def create_clusters(centroids, K, dataset, P):
    m, _ = np.shape(dataset)
    cluster_idx = np.empty(m)
    for i in range(m):
        cluster_idx[i] = closest_centroid(dataset[i], centroids, K, P)
    return cluster_idx


def compute_means(cluster_idx, K, dataset):
    _, n = np.shape(dataset)
    centroids = np.empty((K, n))
    for i in range(K):
        points = dataset[cluster_idx == i]  # gather points for the cluster i
        centroids[i] = np.mean(
            points, axis=0
        )  # use axis=0 to compute means across points
    return centroids


# ================= Main Function ================ #


def run_Kmeans(K, dataset, max_iterations=100, initialization=[], P=np.eye(2)):
    if len(initialization) > 0 and len(initialization) != K:
        raise ValueError("Number of initializations must be equal to K")

    if len(initialization) > 0:
        centroids = np.array(initialization)
    else:
        centroids = initialize_random_centroids(K, dataset)

    for i in range(max_iterations):
        clusters = create_clusters(centroids, K, dataset, P)

        if i % 20 == 0:
            plt.scatter(
                list(dataset[:, 0]) + list([a[0] for a in centroids]),
                list(dataset[:, 1]) + list([a[1] for a in centroids]),
                c=list(clusters) + [K] * 5,
            )
            plt.savefig(
                f"output/{'euclidean' if np.array_equal(P, np.eye(2)) else 'mahalanobis'}_{i // 20}.png"
            )
            plt.clf()  # clear the plot

        centroids = compute_means(clusters, K, dataset)

    return clusters


if __name__ == "__main__":
    # Config
    file_path = "data/f150_motor_distributors.txt"
    output_dir = "output/"
    K = 5
    initializations = np.array([(10, 10), (-10, -10), (2, 2), (3, 3), (-3, -3)])
    P = np.array([[10, 0.5], [-10, 0.25]])  # Covariance matrix P

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load dataset
    dataset = np.loadtxt(file_path, delimiter=",")
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.savefig(f"output/start.png")

    # Run K-means
    y_preds = run_Kmeans(K, dataset, initialization=initializations, P=P)
    plt.clf()
    plt.scatter(dataset[:, 0], dataset[:, 1], c=y_preds)
    plt.savefig(f"output/{'euclidean' if np.array_equal(P, np.eye(2)) else 'mahalanobis'}_result.png")

    # Calculate the first principal component
    pca = PCA(n_components=1)
    pca.fit(dataset)
    first_principal_component = pca.components_[0]

    print("First Principal Component:", first_principal_component, "\n")

    # Calculate the first principal component for each cluster
    for i in set(y_preds):
        pca = PCA(n_components=1)
        pca.fit(dataset[y_preds == i])
        print(f"First Principal Component for Cluster {int(i+1)}:", pca.components_[0])
    

