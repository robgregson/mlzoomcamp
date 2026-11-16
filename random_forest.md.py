
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from constants import *

# --------------------------------------------------
# 1. Load and prepare the data
# --------------------------------------------------

df = pd.read_csv(prepared_data_file_path)


X = df.drop(columns=['default'])
y = df['default']


# --------------------------------------------------
# 2. Split into training, validation and test sets
# --------------------------------------------------

# First split: 60% training, 40% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Second split: 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# --------------------------------------------------
# 3. Define threshold selection function
# --------------------------------------------------

def choose_threshold(y_true, y_prob):

    thresholds = np.arange(0.1, 0.91, 0.01)

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        score = f1_score(y_true, y_pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


# --------------------------------------------------
# 4. Create and train the Random Forest
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)


# --------------------------------------------------
# 5. Select the classification threshold
# --------------------------------------------------

# Get default probabilities for validation data
y_val_prob = random_forest.predict_proba(X_val)[:, 1]

# Select threshold that maximises validation F1
best_threshold = choose_threshold(y_val, y_val_prob)

print("Selected threshold:", best_threshold)

# Show validation performance
y_val_pred = (y_val_prob >= best_threshold).astype(int)

print("\nValidation performance:")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred))
print("Recall:", recall_score(y_val, y_val_pred))
print("F1:", f1_score(y_val, y_val_pred))


# --------------------------------------------------
# 6. Evaluate on the independent test set
# --------------------------------------------------

# Obtain predicted probabilities
y_test_prob = random_forest.predict_proba(X_test)[:, 1]

# Apply the selected threshold
y_test_pred = (y_test_prob >= best_threshold).astype(int)

print("\nTest performance:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1:", f1_score(y_test, y_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_test_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# --------------------------------------------------
# 7. Feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='importance',
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))
