import pandas as pd
import numpy as ny
import matplotlib.pyplot as plt

ny.random.seed(42)


student_numbers = [f'{i:02d}' for i in range(1, 201)]  # This will generate '01', '02', ..., '200'


student_ids = [f'2022-CS-{num}' for num in student_numbers]

names = [
    'Salman', 'Eeman', 'Laiba', 'Aliha', 'Dania', 'Sarah', 'Ayesha', 'Jami', 
    'Fatima', 'Sehar', 'Umaima', 'Minhal', 'Samina', 'Zainab', 'Amna', 
    'irha', 'Hamza', 'Natasha', 'Abdulah', 'Ashhad'
]

Student_Name=ny.random.choice(names,size=200)

Subjects=ny.random.choice(['Artificial Intelligence', 'Computer Networks','Software Engineering','PPSD','DSA'],size=200)

Student_score=ny.random.uniform(0,100,200)

Student_DataFrame=pd.DataFrame({
    'StudentID':student_ids,
    'Name':Student_Name,
    'Subject':Subjects,
    'Score':Student_score
})


# ---------------- TASK 1 -----------
# print(Student_DataFrame.head(10))

# ----------------- TASK 2-----------------

# Student_scores= Student_DataFrame['Score'].to_numpy()

# MeanScore= ny.mean(Student_scores)
# median_score = ny.median(Student_scores)
# std_dev_score = ny.std(Student_scores)

#----------------- TASK 3 ---------------

# FilteredStudents=Student_DataFrame[(Student_DataFrame['Score'] > 80)]


# ---------------- TASK 4 ----------------
GroupedData=Student_DataFrame.groupby('Subject')['Score'].mean()


# ---------------- TASK 5 ---------------
# plt.figure(figsize=(8, 6))  # Set figure size
# plt.hist(Student_DataFrame['Score'], bins=10, edgecolor='black', color='skyblue')

# plt.title('Distribution of Exam Scores', fontsize=16)
# plt.xlabel('Scores', fontsize=12)
# plt.ylabel('Number of Students', fontsize=12)

# plt.show()

plt.figure(figsize=(8,6))
GroupedData.plot(kind='bar', color='skyblue')
plt.title('Average Scores across different subjects')
plt.xlabel('Subjects')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()