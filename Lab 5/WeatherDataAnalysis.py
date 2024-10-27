import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(400)

dates = pd.date_range(start="2023-01-01", periods=365)

temperature = np.random.uniform(10, 40, size=365)
humidity = np.random.uniform(30, 90, size=365)
wind_speed = np.random.uniform(0, 20, size=365)

weather_condition = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=365)

weather_data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind Speed': wind_speed,
    'Weather Condition': weather_condition
})
#    ----   PART A :: 

print(weather_data.head(20))  # Displays the first 10 rows

#   ----------PART B-------------
temperature_array = weather_data['Temperature'].to_numpy()

# mean_temp = np.mean(temperature_array)
# median_temp = np.median(temperature_array)
# std_dev_temp = np.std(temperature_array)

#       ---------------- PART C----------------

# sunny_and_hot_days = weather_data[(weather_data['Temperature'] > 30) & (weather_data['Weather Condition'] == 'Sunny')]

# ------------------- PART D ---------------

# print("Number of days where temperature was above 30°C and it was Sunny: {num_sunny_and_hot_days}")


#------------------------ PART E _----

# avg_humidity_per_condition = weather_data.groupby('Weather Condition')['Humidity'].mean()

# print(avg_humidity_per_condition)


#  ------------- PART F ---------

# plt.figure(figsize=(10, 6))
# plt.plot(weather_data['Date'], weather_data['Temperature'], label='Temperature', color='b')
# plt.title('Temperature Variation Over the Year')
# plt.xlabel('Date')
# plt.ylabel('Temperature (°C)')
# plt.grid(True)
# plt.show()


# ------------ PART G ----------
weather_counts = weather_data['Weather Condition'].value_counts()

plt.figure(figsize=(8, 5))
weather_counts.plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title('Number of Days for Each Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Days')
plt.show()



