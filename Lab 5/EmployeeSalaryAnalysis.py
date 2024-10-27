import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

employee_ids = np.arange(1, 301)

names = [
    'Salman', 'Eeman', 'Laiba', 'Aliha', 'Dania', 'Sarah', 'Ayesha', 'Jami', 
    'Fatima', 'Sehar', 'Umaima', 'Minhal', 'Samina', 'Zainab', 'Amna', 
    'irha', 'Hamza', 'Natasha', 'Abdulah', 'Ashhad'
]

employee_names = np.random.choice(names, size=300)

departments = ['HR', 'IT', 'Finance', 'Sales', 'Marketing']
employee_departments = np.random.choice(departments, size=300)

salaries = np.random.uniform(30000, 120000, size=300)

years_of_experience = np.random.randint(1, 26, size=300)

employee_data = pd.DataFrame({
    'Employee ID': employee_ids,
    'Name': employee_names,
    'Department': employee_departments,
    'Salary': salaries,
    'Years of Experience': years_of_experience
})


# -------------- TASK 1 ---------
# Display the first few rows
# print(employee_data.head(20))


# ------------- TASK 2 -----------

# Create a NumPy array from the Salary column
salary_array = employee_data['Salary'].to_numpy()

# Calculate average, maximum, and minimum salary
average_salary = np.mean(salary_array)
max_salary = np.max(salary_array)
min_salary = np.min(salary_array)



#------------- TASK 3 -------------------

# Filter employees with more than 5 years of experience and above average salary
# high_experience_high_salary = employee_data[
#     (employee_data['Years of Experience'] > 5) & 
#     (employee_data['Salary'] > average_salary)
# ]

# Display the filtered DataFrame
# print(high_experience_high_salary)

#------------ TASK 4 --------------

mean_salary_by_department = employee_data.groupby('Department')['Salary'].mean()

# print(mean_salary_by_department)


# ------------ TASK 5 -------------------

# plt.figure(figsize=(8, 6))
# mean_salary_by_department.plot(kind='bar', color='skyblue')
# plt.title('Average Salary by Department')
# plt.xlabel('Department')
# plt.ylabel('Average Salary ($)')
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.show()



# ---------------- TASK 6 ---------------------

# Create a line plot for salary distribution vs. years of experience
plt.figure(figsize=(10, 6))
plt.plot(employee_data['Years of Experience'], employee_data['Salary'], marker='o', linestyle='-')
plt.title('Salary Distribution vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.grid(True)
plt.show()


