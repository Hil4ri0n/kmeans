import numpy as np


def initialize_centroids_forgy(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    return centroids


def initialize_centroids_kmeans_pp(data, k):
    centroids = [data[np.random.randint(len(data))]]
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(point - centroid) ** 2 for centroid in centroids]) for point in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        random_number = np.random.rand()
        for i, probability in enumerate(cumulative_probabilities):
            if random_number < probability:
                centroids.append(data[i])
                break
    return np.array(centroids)


def assign_to_cluster(data, centroids):
    assignments = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
    return assignments


def update_centroids(data, assignments):
    centroids = np.array([data[assignments == k].mean(axis=0) for k in range(assignments.max() + 1)])
    return centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))


def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

