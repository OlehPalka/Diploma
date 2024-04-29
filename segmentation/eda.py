import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "data/UserTable_optimeal.csv"  # Update the path if needed
data = pd.read_csv(file_path)

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

# Display basic information about the dataset
print("Data Types and Missing Values:")
print(data.info())

# Statistical summary of numerical features
print("\nStatistical Summary of Numerical Features:")
print(data.describe())

for i in data:
    print(data[i].describe())

# Specify a directory to save the plots
save_dir = "segmentation\optimeal_eda_pics"  # Update this to your desired path

# Visualize distributions of numerical features and save the plots
for column in data.select_dtypes(include=["float64", "int64"]).columns:
    plt.figure(figsize=(8, 4))
    plot = sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    # Save the plot
    plt.savefig(f"{save_dir}/Distribution_{column}.png")
    plt.close()

# Box plots to check for outliers and save the plots
for column in data.select_dtypes(include=["float64", "int64"]).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[column])
    plt.title(f"Box Plot of {column}")
    plt.xlabel(column)
    # Save the plot
    plt.savefig(f"{save_dir}/Boxplot_{column}.png")
    plt.close()
