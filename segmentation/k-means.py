from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns

companies = ["optimeal", "club4paws"]

for company in companies:
    # Load the dataset
    file_path = f"data/UserTable_{company}.csv"
    data = pd.read_csv(file_path)

    # Create IsChurned column: True if 'DaysSinceLastPurchase' is greater than 90
    data["IsChurned"] = data["DaysSinceLastPurchase"] > 90

    # Selecting features for clustering
    features = data[
        [
            "AveragePurchaseValue",
            "TotalItemsPurchased",
            "OrdersCount",
            "DaysSinceLastPurchase",
            "PurchaseFreq",
            "OrderSize",
        ]
    ]

    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Identify outliers
    outliers_mask = (np.abs(stats.zscore(features_scaled)) < 3).all(axis=1)

    # Remove churned users and outliers for clustering
    active_not_outliers = (~data["IsChurned"]) & outliers_mask

    # Fit the KMeans model on active users without outliers
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(features_scaled[active_not_outliers])

    # Assign cluster labels to active users without outliers
    data.loc[active_not_outliers, "Cluster"] = kmeans.labels_.astype(int)

    # For churned users and outliers, assign a separate label, such as -1
    data.loc[~active_not_outliers, "Cluster"] = -1

    # Calculate the centroids of the clusters in the original data space
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)

    # Create a DataFrame for the centroids to interpret real-value characteristics
    centroids_df = pd.DataFrame(centroids, columns=features.columns)
    print(centroids_df)

    # Now each active user in the original dataset has a corresponding cluster label,
    # including churned users and outliers which are labeled with -1.
    data = data.drop("LTV", axis=1)
    print(data)

    new_file_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    data.to_csv(new_file_path, index=False)

    # Plotting the clusters
    plt.figure(figsize=(12, 6))
    for i in range(centroids.shape[1]):
        plt.subplot(2, 3, i + 1)
        plt.hist(features_scaled[:, i], bins=20, alpha=0.5, label="Data")
        for j in range(centroids.shape[0]):  # Adjusting this loop
            plt.axvline(
                x=centroids[j, i],
                color="r",
                linestyle="dashed",
                linewidth=2,
                label="Centroid " + str(j),
            )
        plt.title(features.columns[i])
        plt.legend()
    plt.savefig(f"segmentation/clusters_visualised_{company}.png")
    plt.tight_layout()
    plt.show()
