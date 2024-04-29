from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd

companies = ["optimeal", "club4paws"]

for company in companies:
    # Load the dataset
    file_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    data = pd.read_csv(file_path)

    # Assuming 'Cluster' column exists and '-1' is used for outliers/churned users
    # Filter out the outliers/churned users before computing the Davies-Bouldin Index
    filtered_data = data[data["Cluster"] != -1]

    # Selecting features for Davies-Bouldin Index
    features = filtered_data[
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

    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(features_scaled, filtered_data["Cluster"])
    print(f"Davies-Bouldin Index for {company}: {db_index}")
