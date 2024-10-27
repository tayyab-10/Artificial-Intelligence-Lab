import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(40)

Employee_id = np.arange(1, 1501)
Age = np.random.randint(20, 60, size=1500)
Experience = np.random.randint(1, 40, size=1500)
Gender = np.random.choice(["Male", "Female"], size=1500)
PerformanceRating = np.random.randint(1, 5, size=1500)

data_table = pd.DataFrame({
    "Employee ID": Employee_id,
    "Employee Age": Age,
    "Experience": Experience,
    "Gender": Gender,
    "Performance Rating": PerformanceRating
})

# Step 2: 
data_table.loc[np.random.choice(data_table.index, size=50, replace=False), 'Experience'] = np.nan

data_table['Experience'].fillna(data_table['Experience'].median(), inplace=True)



# Step 3: 
data_table['Gender'] = data_table['Gender'].map({"Male": 0, "Female": 1})


# Step 4:
plt.figure(figsize=(8, 6))

plt.boxplot(data_table['Experience'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Boxplot for Years of Experience')
plt.xlabel('Years of Experience')
plt.show()

# IQR for outlier detection
Q1 = data_table['Experience'].quantile(0.25)
Q3 = data_table['Experience'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound, " Upper Bound:", upper_bound)

# Option 1:
data_cleaned = data_table[(data_table['Experience'] >= lower_bound) & (data_table['Experience'] <= upper_bound)]

# Option 2: CAp Outliers
data_table['Experience'] = np.where(data_table['Experience'] > 40, 40, data_table['Experience'])



# Step 6: Feature Scaling using Z-score normalization
scaler = StandardScaler()

columns_to_scale = ['Employee Age', 'Experience']

data_scaled = data_table.copy()
data_table[columns_to_scale] = scaler.fit_transform(data_table[columns_to_scale])

print(data_scaled.head())


#STEP 7:

plt.figure(figsize=(8, 6))
plt.boxplot(data_table['Performance Rating'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Box Plot of Performance Rating')
plt.xlabel('Performance Rating')
plt.show()

# Scatter Plot 
plt.figure(figsize=(8, 6))
plt.scatter(data_table['Experience'], data_table['Performance Rating'], color='green')
plt.title('Scatter Plot: Years of Experience vs. Performance Rating')
plt.xlabel('Years of Experience')
plt.ylabel('Performance Rating')
plt.grid(True)
plt.show()


# STEP 8:
correlation_matrix = data_table[['Age', 'Experience', 'PerformanceRating']].corr()


#  STEP 9:

data_table['Experience Per Age'] = data_table['Experience'] / data_table['Age']


# print(data_table[['Age', 'Annual Income', 'Income per Age']].head())


# Step 10:

data = data_table.drop('Employee ID', axis=1)

print(data.head())

X = data_table.drop('Employee ID', axis=1)  
y = data_table['Employee ID']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape, ",", y_train.shape)
print("Testing set shape:", X_test.shape, ",", y_test.shape)