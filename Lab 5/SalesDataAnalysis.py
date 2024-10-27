import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dates = pd.date_range(end="2023-12-31", periods=500, freq='D')


products = np.random.choice(['iPhone', 'Samsung Galaxy', 'MacBook Pro', 'Dell XPS', 
                             'PlayStation 5', 'Xbox Series X', 'AirPods', 'Kindle', 
                             'GoPro Hero', 'Fitbit Charge'], size=500)

prices = np.random.uniform(10, 1000, size=500)

quantities = np.random.randint(1, 21, size=500)

sales_data = pd.DataFrame({
    'Order ID': np.arange(1, 501),
    'Product': products,
    'Price': prices,
    'Quantity': quantities,
    'Date of Purchase': dates
})


#--------------- TASk 1 ---------------
# Display the first few rows
# print(sales_data.head())


# ------------- TASK 2 ---------------

# price_quantity_array = sales_data[['Price', 'Quantity']].to_numpy()

# total_sales = price_quantity_array[:, 0] * price_quantity_array[:, 1]

# sales_data['Total Sales'] = total_sales

# print(sales_data.head())


# --------------- TASK 3 -------------------

# filtered_sales_data = sales_data[sales_data['Total Sales'] > 100]

# print(filtered_sales_data.head())


# ----------------- TASK 4 ---------------]

# product_quantity_sold = sales_data.groupby('Product')['Quantity'].sum()

# Display the total quantity sold for each product
# print(product_quantity_sold)


# ----------------------- TASK 5 ------------------


# plt.figure(figsize=(8, 6))
# plt.scatter(sales_data['Price'], sales_data['Quantity'], alpha=0.6)
# plt.title('Price vs Quantity of Products Sold')
# plt.xlabel('Price ($)')
# plt.ylabel('Quantity Sold')
# plt.grid(True)
# plt.show()

# ----------------- TASK 6 -------------------

plt.figure(figsize=(8, 6))
plt.hist(sales_data['Total Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



