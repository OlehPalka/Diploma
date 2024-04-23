# Diploma
Diploma Project

Data Description
Orders_Club4paws/Optimeal
• OrderId Unique order identification (Can contain several products).
• Date Date, when the order was created.
• UserId Id of a user.
• ProductId It is a specific product of the store.
• Amount Amount of product ordered.
• Price Total price per specific product.
Chapter 5. Data Description and Preparation 13

Data Collection and Preprocessing

The data was given in CSV format. The first step was removing all the sensitive data,
such as the full user name and surname (and replacing them with user IDs), phone
numbers, and addresses.
Also, I have created two tables with products and their up-to-date information,
two tables of users, and all the useful information about them and features from the
beginning datasets.

ProductTable
• ProductId It is a specific store product.
• Price Current product price.
• FullName Full name of the product displayed on the website.
• ProductPopularity Amount of times it was bought.
• ProductReorderRate For products purchased more than once by the same user,
the reorder rate is calculated to understand which products have repeat pur-
chase potential.
• OftenBoughtWith Id of product which is often bought with this product in one
purchase.

UserTable
• UserId Id of a user.
• DaysSinceLastPurchase For each user, the number of days since their last pur-
chase is calculated to track user engagement over time
• DayOfWeek Day of the week at which the user made most of his orders.
• Hour Hour of the day when the user created most of his orders.
• OrdersCount Amount of orders made.
• TotalSpent Total money spent in the store.
• AveragePurchaseValue Average value of users purchase.
• PurchseFreq Purchase frequency.
• PreferedProduct Product ID of the product the most frequently bought.
• TotalItemsPurchased Amount of products purchased by user.
• OrderSize Average amount of products bought by a user per one order.
• LT User lifetime (time from the first to the last purchase).
• LTV User lifetime value (total spend per user).
