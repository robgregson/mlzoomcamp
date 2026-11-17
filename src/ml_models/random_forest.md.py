import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.data_modelling import load_training_data, list_model_metrics

def choose_best_threshold(y_true, y_prob:np.ndarray):
    # using f1
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    for best_threshold in thresholds:
        y_pred = (y_prob >= best_threshold).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = best_threshold

    return best_threshold

# load the data
X_train, y_train, X_val, y_val, X_test, y_test = load_training_data()

# create and train the random forest model
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

y_val_prob = random_forest.predict_proba(X_val)[:, 1]

# Select threshold that maximises validation F1
best_threshold = choose_best_threshold(y_val, y_val_prob)
print("Selected threshold:", best_threshold)

# Show validation performance
y_val_pred = (y_val_prob >= best_threshold).astype(int)
print("\nValidation performance:")
list_model_metrics(y_val, y_val_pred)

# evaluate on the independent test set
y_test_prob = random_forest.predict_proba(X_test)[:, 1]
# Apply the selected threshold
y_test_pred = (y_test_prob >= best_threshold).astype(int)
list_model_metrics(y_test, y_test_pred, y_test_prob)

# feature importance
feature_importance = pd.DataFrame({'feature': X_train.columns,'importance': random_forest.feature_importances_})
feature_importance = feature_importance.sort_values(by='importance',ascending=False)
print("\nFeature Importance:")
print(feature_importance.to_string(index=False))
# Note: Random Forest feature importance measures impurity reduction, not necessarily the actual predictive contribution
# of a feature. It can favour continuous or high-cardinality features.
