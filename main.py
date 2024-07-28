import numpy as np
from test_folder.k_means import kmeans
from test_folder.k_means import plot_results

if __name__ == "__main__":
    # Create a sample dataset
    num_points = 1000
    X = np.random.rand(num_points, 2)

    # Set number of clusters
    k = 3
                
    # Run K-Means
    centroids, labels = kmeans(X, k)

    # Plot the results
    plot_results(X, centroids, labels)

    # Print the results
    print("Centroids:")
    print(centroids)
    print("Labels:")
    print(labels)