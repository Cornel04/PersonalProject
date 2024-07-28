import numpy as np
from test_folder.k_means import kmeans
from test_folder.k_means import plot_results

if __name__ == "__main__":
    # Create a sample dataset
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], 
                  [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]])

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
