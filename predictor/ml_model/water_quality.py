import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("water_potability.csv")

# Handling missing values using median
df.fillna(df.median(), inplace=True)

# Splitting features and target
X = df.drop(columns=['Potability'])
y = df['Potability']

# Feature Selection using Recursive Feature Elimination (RFE)
rfe = RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=6)
X_selected = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Apply SMOTE for class imbalance
smote = SMOTE(sampling_strategy=0.75, k_neighbors=3, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier()
}

# Hyperparameter tuning
param_grids = {
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "Decision Tree": {'max_depth': [5, 10, 20]},
    "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "XGBoost": {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
}

tuned_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    tuned_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")

# Ensemble Learning - Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', tuned_models['Random Forest']),
    ('svm', tuned_models['SVM']),
    ('xgb', tuned_models['XGBoost'])
], voting='soft')

voting_clf.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Evaluate all models
for name, model in tuned_models.items():
    print(f"\n{name} Performance:")
    evaluate_model(model, X_test, y_test)

# Evaluate ensemble model
print("\nVoting Classifier Performance:")
evaluate_model(voting_clf, X_test, y_test)

import os
import joblib

# Load the model
model_path = os.path.join(os.getcwd(), 'predictor', 'ml_model', 'water_quality_model.pkl')
scaler_path = os.path.join(os.getcwd(), 'predictor', 'ml_model', 'scaler.pkl')
features_path = os.path.join(os.getcwd(), 'predictor', 'ml_model', 'selected_features.pkl')

water_quality_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
selected_features = joblib.load(features_path)

print("Model loaded successfully!")
