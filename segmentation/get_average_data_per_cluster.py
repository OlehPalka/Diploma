import pandas as pd
companies = ["optimeal", "club4paws"]


def get_average_data_per_cluster(company):
    file_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    data = pd.read_csv(file_path)

    columns_to_include = [
        "TotalSpent",
        "LT",
        "AveragePurchaseValue",
        "TotalItemsPurchased",
        "OrdersCount",
        "DaysSinceLastPurchase",
        "PurchaseFreq",
        "OrderSize",
    ]
    return data.groupby("Cluster")[columns_to_include].mean()


for company in companies:
    averages_per_cluster = get_average_data_per_cluster(company)
    print()
    print()
    print(company)
    print(averages_per_cluster)
