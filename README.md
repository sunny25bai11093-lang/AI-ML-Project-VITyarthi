#AIdyn
Student Academic Performance Predictor using Supervised Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#twinkle
data = {
    'Study_Hours': [2, 5, 3, 8, 1, 6, 7, 4, 9, 2, 5, 4],
    'Attendance_Percentage': [70, 85, 75, 95, 60, 88, 92, 80, 98, 72, 84, 78],
    'Previous_Score': [60, 75, 65, 85, 50, 80, 88, 70, 95, 62, 78, 73],
    'Final_Grade': [62, 78, 68, 90, 52, 82, 91, 74, 97, 65, 80, 76]
}

df = pd.DataFrame(data)

#babablacksheep
X = df[['Study_Hours', 'Attendance_Percentage', 'Previous_Score']]
y = df['Final_Grade']

#lov
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#jeal
model = LinearRegression()
model.fit(X_train, y_train)

#hate
predictions = model.predict(X_test)

#pure
print("--- Student Performance Predictor Results ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")
print(f"R-squared Score: {r2_score(y_test, predictions):.2f}")

#startstar
sample_student = [[6, 85, 75]] # 6 hours study, 85% attendance, 75 prev score
pred = model.predict(sample_student)
print(f"\nPredicted Grade for sample student: {pred[0]:.2f}")
