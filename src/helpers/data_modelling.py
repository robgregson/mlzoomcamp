import pandas as pd
from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from src.config.constants import PREPARED_DATA_FILE_PATH

def load_training_data():

    df = pd.read_csv(PREPARED_DATA_FILE_PATH)
    # Separate features and target
    X = df.drop(columns=['default'])
    y = df['default']
    # First split: 60% train, 40% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    # Second split: divide remaining 40% equally
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    return X_train, y_train, X_val, y_val, X_test, y_test


def list_model_metrics(y_true:ndarray, y_predicted:ndarray, y_probabilities:(ndarray|None) = None):

    print("\n")
    print(f"Accuracy (total proportion correctly classified):                   {accuracy_score(y_true, y_predicted):.4f}")
    print(f"Precision (of predicted defaults, how many actually defaulted):     {precision_score(y_true, y_predicted):.4f}")
    print(f"Recall (of defaulters, how many were spotted):                      {recall_score(y_true, y_predicted):.4f}")
    print(f"F1 (precision v recall):                                            {f1_score(y_true, y_predicted):.4f}")
    if y_probabilities is not None:
        print(f"ROC-AUC (how well defaulters distinguished from non-defaulters):    {roc_auc_score(y_true, y_probabilities):.4f}")

    print("\nConfusion Matrix (TP, TN) (FP, FN):")
    print(confusion_matrix(y_true, y_predicted))

    print("\nClassification Report:")
    print(classification_report(y_true, y_predicted))


def list_metrics_for_thresholds(y_values:ndarray, y_val_probabilities:ndarray, thresholds: list[float]) -> None:

    for threshold in thresholds:
        y_pred = (y_val_probabilities >= threshold).astype(int)
        print(f"\nThreshold: {threshold}")
        print(f"Accuracy:  {accuracy_score(y_values, y_pred):.4f}")
        print(f"Precision: {precision_score(y_values, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_values, y_pred):.4f}")
        print(f"F1:        {f1_score(y_values, y_pred):.4f}")
