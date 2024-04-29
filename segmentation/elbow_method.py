import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

companies = ["optimeal", "club4paws"]

for i in companies:
    # Load the dataset
    file_path = f"data/UserTable_{i}.csv"
    data = pd.read_csv(file_path)

    # Select the variables of interest for clustering
    selected_variables = [
        "AveragePurchaseValue",
        "TotalItemsPurchased",
        "OrdersCount",
        "DaysSinceLastPurchase",
        "PurchaseFreq",
        "OrderSize",
    ]

    # Subsetting the dataset
    data_subset = data[selected_variables]

    # Handling outliers - using the interquartile range (IQR) method
    Q1 = data_subset.quantile(0.25)
    Q3 = data_subset.quantile(0.75)
    IQR = Q3 - Q1

    # Define limits for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    data_filtered = data_subset[
        ~((data_subset < lower_bound) | (data_subset > upper_bound)).any(axis=1)
    ]

    # Normalization using standardization (z-score normalization)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_filtered)

    # Implementing the elbow method to find the optimal number of clusters
    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_normalized)
        sse[k] = (
            kmeans.inertia_
        )  # Sum of squared distances of samples to their closest cluster center

    # Plot the elbow graph
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()), marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title(F"Elbow Method to Determine Optimal Number of Clusters of {i}")
    plt.xticks(list(sse.keys()))
    plt.savefig(f"segmentation/clusters_number_{i}.png")
    plt.show()
