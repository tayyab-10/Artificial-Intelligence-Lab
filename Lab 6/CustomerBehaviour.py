import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(42)

customer_id = np.arange(1, 1001)

age = np.random.randint(18, 71, size=1000)

annual_income = np.random.randint(20000, 120001, size=1000)

gender = np.random.choice(['Male', 'Female'], size=1000)

purchased = np.random.choice([0, 1], size=1000)

data = pd.DataFrame({
    'CustomerID': customer_id,
    'Age': age,
    'Annual Income': annual_income,
    'Gender': gender,
    'Purchased': purchased
})

# Display the first few rows of the dataset
# print(data.head())

# STEP 2 :
# Check for missing values
missing_values = data.isnull().sum()
# print(missing_values)


#   STEP 3 :
data.loc[np.random.choice(data.index, size=50, replace=False), 'Annual Income'] = np.nan

# print(data['Annual Income'].isnull().sum())

median_income = data['Annual Income'].median()
data['Annual Income'].fillna(median_income, inplace=True)

# print(data['Annual Income'].isnull().sum())


# STEP 4:

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
# print(data.head())


# STEP 5:

scaler = MinMaxScaler()

# Scale Age and Annual Income columns
data[['Age', 'Annual Income']] = scaler.fit_transform(data[['Age', 'Annual Income']])

# print(data[['Age', 'Annual Income']].head())




# Create a histogram 
plt.hist(data['Age'], bins=10, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# # Create a scatter plot 
plt.scatter(data['Age'], data['Annual Income'], alpha=0.5)
plt.title('Age vs. Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.show()


# Calculate the correlation matrix
correlation_matrix = data[['Age', 'Annual Income', 'Purchased']].corr()

# print(correlation_matrix)

# # STEP 8:

# data['Income per Age'] = data['Annual Income'] / data['Age']

# print(data[['Age', 'Annual Income', 'Income per Age']].head())


# # STEP 9:

data = data.drop('CustomerID', axis=1)

print(data.head())


X = data.drop('Purchased', axis=1)  
y = data['Purchased']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape, ",", y_train.shape)
print("Testing set shape:", X_test.shape, ",", y_test.shape)

