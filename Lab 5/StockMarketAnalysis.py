import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")

companies = np.random.choice(['Google', 'Meta', 'Microsoft', 'Amazon', 'Systems'], size=1000)

open_price = np.random.uniform(50, 500, size=1000)
close_price = np.random.uniform(50, 500, size=1000)

volume_traded = np.random.randint(1000, 1000000, size=1000)

stock_data = pd.DataFrame({
    'Date': dates,
    'Company': companies,
    'Open Price': open_price,
    'Close Price': close_price,
    'Volume Traded': volume_traded
})

#--------------- TASK 1 ------------------
print(stock_data.head(10))

# ---------------- TASK 2 ----------------

close_price_array = stock_data['Close Price'].to_numpy()

daily_percentage_change = np.diff(close_price_array) / close_price_array[:-1] * 100

stock_data['Percentage Change'] = np.insert(daily_percentage_change, 0, np.nan)

print(stock_data.head())


# ---------- TASK 3 -------------------

increased_days = stock_data[stock_data['Percentage Change'] > 2]

# ----------------- TASK 4 ------------------

total_volume_traded = stock_data.groupby('Company')['Volume Traded'].sum()

# ------------ TASK 5 --------------

company_a_data = stock_data[stock_data['Company'] == 'Google']

# Plot the Close Price trend for Company A
plt.figure(figsize=(10, 5))
plt.plot(company_a_data['Date'], company_a_data['Close Price'], label='Close Price')
plt.title('Close Price Trend for Google')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------- TASK 6 ------------

avg_percentage_change = stock_data.groupby('Company')['Percentage Change'].mean()

plt.figure(figsize=(8, 5))
avg_percentage_change.plot(kind='bar', color='skyblue')
plt.title('Average Percentage Change in Close Price by Company')
plt.xlabel('Company')
plt.ylabel('Average Percentage Change')
plt.tight_layout()
plt.show()




