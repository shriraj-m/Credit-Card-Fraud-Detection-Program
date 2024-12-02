import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report, precision_score, f1_score
from app.models.neuralnetwork import FraudDetectionModel
from collections import Counter


def train_and_save_models():
    # Create directories if they don't exist
    os.makedirs('app/models/saved_models', exist_ok=True)
    
    # Load and process data
    df = pd.read_csv("dataset/creditcard.csv")
    
    sns.countplot(x='Class', data=df)
    plt.title("Class Distribution")
    # plt.show()

    X = df.drop(columns = ['Class'])
    y = df['Class']

    feature_names = X.columns.tolist()
    with open('app/models/saved_models/feature_names.txt', 'w') as file:
        file.write(','.join(feature_names))

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    torch.save(scalar, 'app/models/saved_models/scaler.pth')

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    class_counts = Counter(y_train)
    non_fraud = class_counts[0]
    fraud = class_counts[1]
    spw = non_fraud / fraud

    # XGBoost
    xgboost_summary = {}
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        objective='binary:logistic',
        scale_pos_weight=spw, # 580:1 ratio of non-fraud to fraud
        use_label_encoder=False
    )

    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_model.save_model('app/models/saved_models/xgboost_model.json')
    xgb_time = time.time() - start_time

    print(f"XGBoost Training Time: {xgb_time:.2f} seconds")
    xgb_prediction = xgb_model.predict(X_test)
    xgboost_summary["Model"] = "XGBoost"
    print(f"XGBoost Classification Report: \n{classification_report(y_test, xgb_prediction)}")
    xgboost_summary["Classification Report"] = classification_report(y_test, xgb_prediction)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_prediction)}")
    xgboost_summary["Accuracy"] = accuracy_score(y_test, xgb_prediction)
    print(f"XGBoost Recall: {recall_score(y_test, xgb_prediction)}")
    xgboost_summary["Recall"] = recall_score(y_test, xgb_prediction)
    print(f"XGBoost Precision: {precision_score(y_test, xgb_prediction)}")
    xgboost_summary["Precision"] = precision_score(y_test, xgb_prediction)
    print(f"XGBoost F1 Score: {f1_score(y_test, xgb_prediction)}")
    xgboost_summary["F1 Score"] = f1_score(y_test, xgb_prediction)

    input_text = f"""
        Summarize the results of the XGBoost model:
        The Model Used is: {xgboost_summary["Model"]}
        This is the Classification Report: {xgboost_summary["Classification Report"]}
        The Model's Accuracy is: {xgboost_summary["Accuracy"]:.2f}
        The Model's Recall is: {xgboost_summary["Recall"]:.2f}
        The Model's Precision is: {xgboost_summary["Precision"]:.2f}
        The Model's F1 Score is: {xgboost_summary["F1 Score"]:.2f}
    """

if __name__ == "__main__":
    train_and_save_models() 