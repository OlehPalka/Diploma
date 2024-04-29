from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Assuming 'data' is your DataFrame and 'features' are your selected columns for clustering.
pathes = [
    "data/UserTable_optimeal.csv",
    "data/UserTable_club4paws.csv",
]  # Make sure to use your actual file path here
for file_path in pathes:
    data = pd.read_csv(file_path)

    # Define the features you want to use for clustering
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

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Assuming you have already determined the optimal number of clusters 'n_clusters'
    n_clusters = 4  # replace with your optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    # Compute the Calinski-Harabasz score
    calinski_harabasz_idx = calinski_harabasz_score(features_scaled, labels)
    print(f"Calinski-Harabasz Index for your dataset: {calinski_harabasz_idx}")
