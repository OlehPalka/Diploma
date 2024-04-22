import pandas as pd
from datetime import datetime

def process_csv_for_diploma(file_path, store):
    df = pd.read_csv(file_path,  delimiter=';')
    df['UserId'] = pd.factorize(df['Phone Number'])[0] + 1
    df['ProductId'] = pd.factorize(df['Taste'])[0] + 1
    df.drop(columns=['Name', 'Surname', 'Phone Number', 'Adress', 'Promo'], inplace=True)
    df.to_csv(f'processed_{store}_orders.csv', index=False)
    return f'processed_{store}_orders.csv'

def create_product_table(file_path, store):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    latest_prices = df.sort_values('Date').groupby('ProductId').last()['Price']
    product_popularity = df['ProductId'].value_counts()
    reorder_count = df.groupby(['ProductId', 'UserId']).size().reset_index(name='Counts')
    reorder_rate = reorder_count[reorder_count['Counts'] > 1].groupby('ProductId').size()

    order_groups = df.groupby('OrderId')['ProductId'].apply(list)
    product_pairs = []

    for products in order_groups:
        if len(products) > 1:
            for product in set(products):
                co_purchased = [co for co in products if co != product]
                product_pairs.extend([(product, co_product) for co_product in co_purchased])

    pairs_df = pd.DataFrame(product_pairs, columns=['ProductId', 'OftenBoughtWith'])
    often_bought_with = pairs_df.groupby('ProductId')['OftenBoughtWith'].agg(lambda x: x.value_counts().idxmax())

    product_table = pd.DataFrame({
        'ProductId': product_popularity.index,
        'FullName': df.groupby('ProductId').last()['Taste'],
        'Price': latest_prices,
        'ProductPopularity': product_popularity,
        'ProductReorderRate': reorder_rate,
        'OftenBoughtWith': often_bought_with
    }).reset_index(drop=True)

    product_table.to_csv(f'ProductTable_{store}.csv', index=False)
    return f'ProductTable_{store}.csv'

def create_user_table(file_path, store):
    df = pd.read_csv(file_path, dtype={'Price': float}, decimal=',')
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
    order_details = df.drop_duplicates(subset='OrderId')

    order_group = order_details.groupby('UserId')
    item_group = df.groupby(['UserId', 'OrderId'])

    user_table = pd.DataFrame()
    user_table['UserId'] = order_group['UserId'].max()
    user_table['DaysSinceLastPurchase'] = (df['Date'].max() - order_group['Date'].max()).dt.days
    user_table['DayOfWeek'] = order_group['Date'].agg(lambda x: x.dt.day_name().mode()[0])
    user_table['Hour'] = order_group['Date'].agg(lambda x: x.dt.hour.mode()[0])
    user_table['OrdersCount'] = order_group.size()
    user_table['TotalSpent'] = item_group['Price'].sum().groupby('UserId').sum()
    user_table['AveragePurchaseValue'] = user_table['TotalSpent'] / user_table['OrdersCount']
    user_table['PurchaseFreq'] = user_table['OrdersCount'] / ((order_group['Date'].max() - order_group['Date'].min()).dt.days / 30 + 1)
    user_table['PreferedProduct'] = df.groupby('UserId')['ProductId'].agg(lambda x: x.value_counts().idxmax())
    user_table['TotalItemsPurchased'] = df.groupby('UserId')['Amount'].sum()
    user_table['OrderSize'] = df.groupby('UserId')['Amount'].mean()
    user_table['LT'] = (order_group['Date'].max() - order_group['Date'].min()).dt.days
    user_table['LTV'] = user_table['TotalSpent'] 

    user_table.to_csv(f'UserTable_{store}.csv', index=False)
    return f'UserTable_{store}.csv'


if __name__ == "__main__":
    stores = [["optimeal", "optimeal_orders.csv"], ["club4paws", "club4paws_orders.csv"]]
    for store in stores:
        processed_file = process_csv_for_diploma(store[1], store[0])
        create_product_table(processed_file, store[0])
        create_user_table(processed_file, store[0])


