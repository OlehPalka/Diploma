import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

file_path = "data/UserTable_club4paws.csv"
data = pd.read_csv(file_path, delimiter=",")

day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7,
}

data["DayOfWeek"] = data["DayOfWeek"].map(day_mapping)

columns_to_include = [
    "DaysSinceLastPurchase",
    "DayOfWeek",
    "Hour",
    "OrdersCount",
    "TotalSpent",
    "AveragePurchaseValue",
    "PurchaseFreq",
    "PreferedProduct",
    "TotalItemsPurchased",
    "OrderSize",
    "LT",
]

data = data[columns_to_include]
correlation_matrix = data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of User Data")
plt.show()
