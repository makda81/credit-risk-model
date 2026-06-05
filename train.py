"""
Model training for credit risk classification.
Trains Logistic Regression and Random Forest, logs to MLflow, and selects the best model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import joblib
import os

# --------------------------------
# 1. Load data
# --------------------------------
print("Loading data...")
df = pd.read_csv('data/processed/customer_risk.csv')
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['is_high_risk']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Scale features (needed for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use in API
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# --------------------------------
# 2. Set up MLflow
# --------------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CreditRisk_Experiment")

# --------------------------------
# 3. Logistic Regression with hyperparameter tuning
# --------------------------------
print("\n--- Logistic Regression ---")
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200]
}
lr = LogisticRegression(random_state=42, class_weight='balanced')
random_search_lr = RandomizedSearchCV(lr, param_grid_lr, n_iter=10, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
random_search_lr.fit(X_train_scaled, y_train)

best_lr = random_search_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
y_proba_lr = best_lr.predict_proba(X_test_scaled)[:, 1]

metrics_lr = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_proba_lr)
}
print("Best params:", random_search_lr.best_params_)
print("Metrics:", metrics_lr)

with mlflow.start_run(run_name="LogisticRegression_Tuned"):
    mlflow.log_params(random_search_lr.best_params_)
    mlflow.log_metrics(metrics_lr)
    mlflow.sklearn.log_model(best_lr, "model")

# --------------------------------
# 4. Random Forest with hyperparameter tuning
# --------------------------------
print("\n--- Random Forest ---")
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
random_search_rf = RandomizedSearchCV(rf, param_grid_rf, n_iter=10, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
random_search_rf.fit(X_train, y_train)  # no scaling needed

best_rf = random_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

metrics_rf = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_proba_rf)
}
print("Best params:", random_search_rf.best_params_)
print("Metrics:", metrics_rf)

with mlflow.start_run(run_name="RandomForest_Tuned"):
    mlflow.log_params(random_search_rf.best_params_)
    mlflow.log_metrics(metrics_rf)
    mlflow.sklearn.log_model(best_rf, "model")

# --------------------------------
# 5. Select best model (by ROC-AUC)
# --------------------------------
if metrics_lr['roc_auc'] >= metrics_rf['roc_auc']:
    best_model = best_lr
    best_model_name = "LogisticRegression"
else:
    best_model = best_rf
    best_model_name = "RandomForest"

print(f"\nBest model: {best_model_name} with ROC-AUC = {max(metrics_lr['roc_auc'], metrics_rf['roc_auc']):.4f}")

# Save best model locally
joblib.dump(best_model, 'models/best_model.pkl')
print("Best model saved to 'models/best_model.pkl'")

# Save comparison summary
summary = pd.DataFrame([metrics_lr, metrics_rf], index=['LogisticRegression', 'RandomForest'])
summary.to_csv('models/model_comparison.csv')
print("Model comparison saved to 'models/model_comparison.csv'")