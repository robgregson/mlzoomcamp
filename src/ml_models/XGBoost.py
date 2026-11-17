import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from src.data_modelling import load_training_data, list_model_metrics

def choose_best_threshold(y_true, y_prob):
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


# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_training_data()

# Account for class imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("Scale pos weight:", scale_pos_weight)


# Create and train XGBoost model
xgboost = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgboost.fit(X_train, y_train)

# Get validation probabilities
y_val_prob = xgboost.predict_proba(X_val)[:, 1]

# --------------------------------------------------
# 1. Validation performance: default threshold

y_val_pred_default = (y_val_prob >= 0.5).astype(int)
print("\nValidation performance (threshold 0.5):")
list_model_metrics(y_val, y_val_pred_default)

# --------------------------------------------------
# 2. Validation performance: optimised threshold

best_threshold = choose_best_threshold(y_val, y_val_prob)
y_val_pred_optimised = (y_val_prob >= best_threshold).astype(int)
print(f"\nOptimised threshold: {best_threshold:.2f}")
print("\nValidation performance (optimised threshold):")
list_model_metrics(y_val, y_val_pred_optimised)

# --------------------------------------------------
# 3. Independent test evaluation

y_test_prob = xgboost.predict_proba(X_test)[:, 1]
# Default threshold
y_test_pred_default = (y_test_prob >= 0.5).astype(int)
print("\nTest performance (threshold 0.5):")
list_model_metrics(y_test, y_test_pred_default, y_test_prob)
# Optimised threshold
y_test_pred_optimised = (y_test_prob >= best_threshold).astype(int)
print(f"\nTest performance (threshold {best_threshold:.2f}):")
list_model_metrics(y_test, y_test_pred_optimised, y_test_prob)

# --------------------------------------------------
# 4. Feature importance

scores = xgboost.get_booster().get_score(importance_type='gain')
feature_importance = pd.DataFrame([
    {'feature': feature, 'importance': score}
    for feature, score in scores.items()
])
feature_importance = feature_importance.sort_values(by='importance',ascending=False)
print("\nFeature Importance (Gain):")
print(feature_importance.to_string(index=False))
