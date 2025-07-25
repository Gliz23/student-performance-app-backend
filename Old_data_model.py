import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load the CSV file
data = pd.read_csv('student_data.csv')

# Print column names and missing values
print("Column names:", data.columns.tolist())
print("Missing values:\n", data.isnull().sum())

# Check unique values in ordinal columns for debugging
ordinal_cols = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 
                'Family_Income', 'Teacher_Quality', 'Distance_from_Home']
for col in ordinal_cols:
    print(f"Unique values in {col}:", data[col].unique())

# Define features and target
X = data.drop(columns=["Exam_Score"])
y = data["Exam_Score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical columns
onehot_cols = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Peer_Influence', 
               'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 
                'Physical_Activity']

# Define category lists for ordinal columns (adjust based on your data)
ordinal_categories = {
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    ' ': ['Low', 'Medium', 'High'],
    'Family_Income': ['Low', 'Medium', 'High'],
    'Teacher_Quality': ['Low', 'Medium', 'High'],
    'Distance_from_Home': ['Near', 'Moderate', 'Far']  # Updated based on error
}

# Preprocessing pipeline with imputation and custom categories
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OrdinalEncoder(categories=[ordinal_categories[col] for col in ordinal_cols]))
        ]), ordinal_cols),
        ('onehot', Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(drop='first', sparse_output=False))
        ]), onehot_cols),
        ('num', Pipeline(steps=[
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ]), numeric_cols)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")

# Save the trained model
joblib.dump(model, 'exam_score_predictor.pkl')
print("Model saved as exam_score_predictor.pkl")