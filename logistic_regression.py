from constants import *

import pandas as pd
import numpy as np
import pandas.api.types as pdt
# import seaborn as sns
# from matplotlib import pyplot as plt
# %matplotlib inline
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(y_true, y_pred, y_proba):

    print("\n")
    print(f"Accuracy (total proportion correctly classified):                   {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (of predicted defaults, how many actually defaulted):     {precision_score(y_true, y_pred):.4f}")
    print(f"Recall (of defaulters, how many were spotted):                      {recall_score(y_true, y_pred):.4f}")
    print(f"F1 (precision v recall):                                            {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC (how well defaulters distinguished from non-defaulters):    {roc_auc_score(y_true, y_proba):.4f}")

    print("\nConfusion Matrix (TP, TN) (FP, FN):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

df = pd.read_csv(prepared_data_file_path)

# print(df.columns)

# Separate features and target
X = df.drop(columns=['default'])
y = df['default']

# First split: 60% train, 40% remaining
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Second split: divide remaining 40% equally
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Crate the model
# Using a pipeline, because:
# - StandardScaler standardises numerical features.
# - Logistic regression benefits from features being on comparable scales.
# - The scaler is fitted only against the training data.
# - The same transformation is automatically applied to validation and test data.

logistic_regression = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000,random_state=42, class_weight='balanced'))
])

logistic_regression.fit(X_train, y_train)

y_val_pred = logistic_regression.predict(X_val)                     # returns 0 or 1
y_val_proba = logistic_regression.predict_proba(X_val)[:, 1]        # returns the probability (good for default prediction)


evaluate_model(y_val, y_val_pred, y_val_proba)

for threshold in [0.5, 0.4, 0.3, 0.2]:

    y_pred = (y_val_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"F1:        {f1_score(y_val, y_pred):.4f}")