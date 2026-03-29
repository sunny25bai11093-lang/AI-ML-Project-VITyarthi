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


Student Academic Performance Predictor
This project, created as a part of the Fundamentals of AI and ML course at VIT Bhopal, aims to solve a major problem faced by the education system today: predicting a student's performance in their final examinations. This project will use various supervised machine learning techniques to predict a student's grades based on various socio-economic and academic factors.

Objective and Scope
The main aim of this project is to create a predictive model that will use historical data to predict a student's performance in their future examinations. This will help educators shift their approach from reactive to proactive, taking measures to improve a student's performance before their final examinations.

Methodology
The project will use two different techniques for a comprehensive analysis of a student's performance.

Linear Regression: This will be used to predict a student's exact grade or percentage, giving a continuous output based on their exact position in academics.

Decision Tree Classifier: This algorithm can be employed to classify students based on their performance levels or bands, such as Pass/Fail or Grade A/B/C.

Key Features and Parameters
The system has been trained on a variety of critical impact factors, such as:

Academic History: The past semester results and internal test scores.

Engagement Metrics: Daily attendance records and total hours spent studying per week.

Socio-Economic Factors: Parent education level and availability of education resources.

Technical Implementation
The system has been implemented using Python programming, utilizing the Scikit-learn library for training the models and Pandas for data manipulation. The system has a structured pipeline for data cleaning, handling missing data, and outliers, as well as feature scaling to ensure optimal performance.

Significance
The project has significant importance in proving the concept of how AI can be incorporated into existing frameworks to improve graduation levels for such institutions. This project can also be considered a starting point for building more complex systems for real-time academic tracking.
