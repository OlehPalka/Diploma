import pandas as pd
import time
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats


companies = ["optimeal", "club4paws"]
company = companies[0]
stores = [
    ["optimeal", "confid_info/optimeal_orders.csv"],
    ["club4paws", "confid_info/club4paws_orders.csv"],
]
store = stores[0]

def process_csv_for_diploma(file_path, store):
    df = pd.read_csv(file_path, delimiter=";")
    df["UserId"] = pd.factorize(df["Phone Number"])[0] + 1
    df["ProductId"] = pd.factorize(df["Taste"])[0] + 1
    df.drop(
        columns=["Name", "Surname", "Phone Number", "Adress", "Promo"], inplace=True
    )
    df.to_csv(f"processed_{store}_orders.csv", index=False)
    return f"processed_{store}_orders.csv"


def create_product_table(file_path, store):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    latest_prices = df.sort_values("Date").groupby("ProductId").last()["Price"]
    product_popularity = df["ProductId"].value_counts()
    reorder_count = (
        df.groupby(["ProductId", "UserId"]).size().reset_index(name="Counts")
    )
    reorder_rate = (
        reorder_count[reorder_count["Counts"] > 1].groupby("ProductId").size()
    )

    order_groups = df.groupby("OrderId")["ProductId"].apply(list)
    product_pairs = []

    for products in order_groups:
        if len(products) > 1:
            for product in set(products):
                co_purchased = [co for co in products if co != product]
                product_pairs.extend(
                    [(product, co_product) for co_product in co_purchased]
                )

    pairs_df = pd.DataFrame(product_pairs, columns=["ProductId", "OftenBoughtWith"])
    often_bought_with = pairs_df.groupby("ProductId")["OftenBoughtWith"].agg(
        lambda x: x.value_counts().idxmax()
    )

    product_table = pd.DataFrame(
        {
            "ProductId": product_popularity.index,
            "FullName": df.groupby("ProductId").last()["Taste"],
            "Price": latest_prices,
            "ProductPopularity": product_popularity,
            "ProductReorderRate": reorder_rate,
            "OftenBoughtWith": often_bought_with,
        }
    ).reset_index(drop=True)

    product_table.to_csv(f"ProductTable_{store}.csv", index=False)
    return f"ProductTable_{store}.csv"


def create_user_table(file_path, store):
    df = pd.read_csv(file_path, dtype={"Price": float}, decimal=",")

    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M:%S")
    order_details = df.drop_duplicates(subset="OrderId")

    order_group = order_details.groupby("UserId")
    item_group = df.groupby(["UserId", "OrderId"])

    user_table = pd.DataFrame()
    user_table["UserId"] = order_group["UserId"].max()
    user_table["DaysSinceLastPurchase"] = (
        df["Date"].max() - order_group["Date"].max()
    ).dt.days
    user_table["DayOfWeek"] = order_group["Date"].agg(
        lambda x: x.dt.day_name().mode()[0]
    )
    user_table["Hour"] = order_group["Date"].agg(lambda x: x.dt.hour.mode()[0])
    user_table["OrdersCount"] = order_group.size()
    user_table["TotalSpent"] = item_group["Price"].sum().groupby("UserId").sum()
    user_table["AveragePurchaseValue"] = (
        user_table["TotalSpent"] / user_table["OrdersCount"]
    )
    user_table["PurchaseFreq"] = user_table["OrdersCount"] / (
        (order_group["Date"].max() - order_group["Date"].min()).dt.days / 30 + 1
    )
    user_table["PreferedProduct"] = df.groupby("UserId")["ProductId"].agg(
        lambda x: x.value_counts().idxmax()
    )
    user_table["TotalItemsPurchased"] = df.groupby("UserId")["Amount"].sum()
    user_table["OrderSize"] = df.groupby("UserId")["Amount"].mean()
    user_table["LT"] = (order_group["Date"].max() - order_group["Date"].min()).dt.days
    user_table["LTV"] = user_table["TotalSpent"]

    user_table.to_csv(f"UserTable_{store}.csv", index=False)
    return f"UserTable_{store}.csv"


def run_k_means_to_add_new_users_with_clusters(company):
    file_path = f"data/UserTable_{company}.csv"
    data = pd.read_csv(file_path)

    data["IsChurned"] = data["DaysSinceLastPurchase"] > 90

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

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    outliers_mask = (np.abs(stats.zscore(features_scaled)) < 3).all(axis=1)
    active_not_outliers = (~data["IsChurned"]) & outliers_mask
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(features_scaled[active_not_outliers])
    data.loc[active_not_outliers, "Cluster"] = kmeans.labels_.astype(int)
    data.loc[~active_not_outliers, "Cluster"] = -1
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids, columns=features.columns)
    data = data.drop("LTV", axis=1)

    new_file_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    data.to_csv(new_file_path, index=False)


def create_dataset_of_subscription_proposals_sent(company):
    # Load the dataset
    file_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    data = pd.read_csv(file_path)

    # Create a new DataFrame with only UserId and IsChurned
    new_data = data[["UserId", "IsChurned", "Cluster"]].copy()

    # Add new columns with default values
    new_data["SubsPlanSent"] = False
    new_data["SubsPlanAccepted"] = False
    new_data["SubsPlanProposed"] = 0
    new_data["SubsPlanChosen"] = 0
    new_data["IsSubsPlanActive"] = 0
    new_data["DateLastSubsProposal"] = "2000-01-01"

    output_file_path = f"data/UserSubscription_{company}.csv"
    new_data.to_csv(output_file_path, index=False)
create_dataset_of_subscription_proposals_sent(company)

def update_user_subscription(company):
    # Load existing subscription data
    subscription_path = f"data/UserSubscription_{company}.csv"
    subscription_data = pd.read_csv(subscription_path)

    # Load user table data
    user_table_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    user_data = pd.read_csv(user_table_path)

    # Identify new users not present in the subscription data
    new_users = user_data[~user_data["UserId"].isin(subscription_data["UserId"])]

    # Select required columns and add default values for the new subscription-related columns
    new_users = new_users[["UserId", "IsChurned", "Cluster"]].copy()
    new_users["SubsPlanSent"] = False
    new_users["SubsPlanAccepted"] = False
    new_users["SubsPlanProposed"] = ""
    new_users["SubsPlanChosen"] = ""
    new_users["IsSubsPlanActive"] = 0
    new_users["DateLastSubsProposal"] = "2000-01-01"

    # Append these new users to the existing subscription data
    updated_subscription_data = pd.concat(
        [subscription_data, new_users], ignore_index=True
    )

    # Update the IsChurned data for all users based on the latest user data
    updated_subscription_data = updated_subscription_data.set_index("UserId")
    user_data = user_data.set_index("UserId")
    updated_subscription_data["IsChurned"] = user_data["IsChurned"]

    # Save updated data back to CSV
    updated_subscription_data.to_csv(subscription_path, index=False)
    print(f"Updated data saved to {subscription_path}")


def reset_subscription_plans(company):
    # Load the dataset
    subscription_path = f"data/UserSubscription_{company}.csv"
    data = pd.read_csv(subscription_path)

    # Convert 'DateLastSubsProposal' from string to datetime for comparison
    data["DateLastSubsProposal"] = pd.to_datetime(data["DateLastSubsProposal"])

    # Get today's date
    today = datetime.now()

    # Define the conditions for resetting the subscription data
    condition = (
        (data["SubsPlanSent"] == True)
        & (data["SubsPlanAccepted"] == False)
        & ((today - data["DateLastSubsProposal"]).dt.days > 30)
    )

    # Update rows where the condition is True
    data.loc[condition, "SubsPlanSent"] = False
    data.loc[condition, "SubsPlanAccepted"] = False
    data.loc[condition, "SubsPlanProposed"] = ""
    data.loc[condition, "SubsPlanChosen"] = ""
    data.loc[condition, "IsSubsPlanActive"] = 0
    data.loc[condition, "DateLastSubsProposal"] = "2000-01-01"

    # Save the updated data back to the CSV
    data.to_csv(subscription_path, index=False)
    print(f"Data updated and saved back to {subscription_path}")


def update_subscription_proposed_plans(company, users_who_get_subs_prop):
    # Load existing subscription data
    subscription_path = f"data/UserSubscription_{company}.csv"
    subscription_data = pd.read_csv(subscription_path)

    # Update the subscription data based on the dictionary entries
    for user_id, (current_date, cluster, subscription_proposed) in users_who_get_subs_prop.items():
        # Find the index(es) of the user_id in the DataFrame
        user_indexes = subscription_data[subscription_data["UserId"] == user_id].index
        # Update the specified columns
        subscription_data.loc[user_indexes, "SubsPlanSent"] = True
        subscription_data.loc[user_indexes, "SubsPlanProposed"] = cluster
        subscription_data.loc[user_indexes, "DateLastSubsProposal"] = current_date
        subscription_data.loc[user_indexes, "SubsPlanProposed"] = subscription_proposed

    # Save the updated data back to CSV
    subscription_data.to_csv(subscription_path, index=False)
    print(f"Updated data saved to {subscription_path}")


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

    grouped_data = data.groupby("Cluster")[columns_to_include].mean()
    result = grouped_data.to_dict(orient="index")
    result = {int(cluster): metrics for cluster, metrics in result.items()}

    return result


def get_user_data_by_id(company, user_id_list):
    file_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    data = pd.read_csv(file_path)
    result = {}
    for user_id in user_id_list:
        user_data = data[data["UserId"] == user_id]
        result[user_id] = user_data.to_dict(orient="records")[0]
    return result


def get_user_ids_active_without_subscription(company):
    data = pd.read_csv(f"data/UserSubscription_{company}.csv")
    filtered_data = data[(data["IsChurned"] == False) & (data["SubsPlanSent"] == False)]
    user_ids = filtered_data["UserId"].tolist()
    return user_ids


def get_user_ids_active_with_subscription_proposal_sent_not_accepted(company):
    data = pd.read_csv(f"data/UserSubscription_{company}.csv")
    filtered_data = data[
        (data["IsChurned"] == False)
        & (data["SubsPlanSent"] == True)
        & (data["SubsPlanAccepted"] == False)
    ]
    user_ids = filtered_data["UserId"].tolist()
    return user_ids


def create_subscription_proposition(user_id, user_data, company):
    # Load necessary data
    top_products_by_cluster = get_top_products_by_cluster(company)
    average_data_per_cluster = get_average_data_per_cluster(company)
    orders_path = f"data/processed_{company}_orders.csv"
    orders_data = pd.read_csv(orders_path)

    # Filter orders for the given user_id
    user_orders = orders_data[orders_data["UserId"] == user_id]

    # Determine the user's cluster
    user_cluster = user_data["Cluster"]

    if not user_orders.empty:
        last_order = user_orders.loc[
            pd.to_datetime(user_orders["Date"], format="%d/%m/%Y %H:%M:%S").idxmax()
        ]
        last_order_id = last_order["OrderId"]

        # Get product IDs from the last order and convert to list
        products_in_last_order = list(
            user_orders[user_orders["OrderId"] == last_order_id]["ProductId"].unique()
        )
        initial_product_count = len(products_in_last_order)
    else:
        products_in_last_order = []
        initial_product_count = 0

    # Cluster average order size
    cluster_average_order_size = average_data_per_cluster[user_cluster]["OrderSize"]
    if initial_product_count < cluster_average_order_size:
        top_cluster_products = top_products_by_cluster.get(user_cluster, [])
        additional_products = [
            prod for prod in top_cluster_products if prod not in products_in_last_order
        ]

        for i in additional_products:
            products_in_last_order.append(i)
            if len(products_in_last_order) > cluster_average_order_size:
                break
        print(products_in_last_order)


    products_in_last_order = (
        ", ".join(map(str, products_in_last_order))
        if products_in_last_order
        else "no products found"
    )

    # Determine the subscription frequency rounded to the nearest 0.5
    frequency = round(user_data["PurchaseFreq"] * 2) / 2

    # Formulate the subscription proposition message
    proposition = f"UserId {user_id} was proposed with the next subscription: every {frequency} month(s), customer would receive the order of these products ({products_in_last_order}) based on his previous orders and clusters most popular products."
    print(proposition)
    return proposition


def get_top_products_by_cluster(company):
    file_path = f"data/UserTable_{company}_clustered_kmeans.csv"
    data = pd.read_csv(file_path)
    product_counts = data.groupby(['Cluster', 'PreferedProduct']).size().reset_index(name='Count')
    top_products = product_counts.sort_values(['Cluster', 'Count'], ascending=[True, False])
    top_5_products_per_cluster = top_products.groupby('Cluster').head(5)
    top_5_dict = (
        top_5_products_per_cluster.groupby("Cluster")["PreferedProduct"]
        .apply(list)
        .to_dict()
    )

    return top_5_dict


# while True:
#     processed_file = process_csv_for_diploma(store[1], store[0])
#     create_product_table(processed_file, store[0])
#     create_user_table(processed_file, store[0])
#     run_k_means_to_add_new_users_with_clusters(company)
#     update_user_subscription(company)

#     active_users_without_subscription = get_user_ids_active_without_subscription(company)
#     active_users_without_subscription_info = get_user_data_by_id(company, active_users_without_subscription)

#     # While loop for intraday proposals sending

#     start_date = datetime.date.today()
#     users_who_get_subs_prop = {}
#     while datetime.date.today() == start_date:

#         current_hour = datetime.datetime.now().hour
#         current_day_of_week = datetime.datetime.now().strftime("%A")

#         for user_id, user_data in active_users_without_subscription_info.items():
#             if (
#                 user_data["DayOfWeek"] == current_day_of_week
#                 and user_data["Hour"] == current_hour
#             ):
#                 print(f"Subscription proposal sent to UserId {user_id}")
#                 users_who_get_subs_prop[user_id] = [
#                     datetime.date.today().strftime("%Y-%m-%d"),
#                     user_data["Cluster"],
#                     create_subscription_proposition(user_id, user_data, company)
#                 ]

#         time.sleep(3600)
#     update_subscription_proposed_plans(company, users_who_get_subs_prop)
#     reset_subscription_plans(company)
