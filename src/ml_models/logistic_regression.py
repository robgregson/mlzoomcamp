from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.helpers.data_modelling import load_training_data, list_model_metrics, list_metrics_for_thresholds

X_train, y_train, X_val, y_val, X_test, y_test = load_training_data()

# Create the model
# Using a pipeline, because:
# - StandardScaler standardises numerical features.
# - Logistic regression benefits from features being on comparable scales.
# - The scaler is fitted only against the training data.
# - The same transformation is automatically applied to validation and test data.
logistic_regression = Pipeline([
    ('scaler', StandardScaler()),
    # ('model', LogisticRegression(max_iter=1000,random_state=42, class_weight='balanced'))
    ('model', LogisticRegression(max_iter=1000,random_state=42))
])
logistic_regression.fit(X_train, y_train)

y_val_pred:ndarray = logistic_regression.predict(X_val)                     # returns 0 or 1
y_val_proba:ndarray = logistic_regression.predict_proba(X_val)[:, 1]        # returns the probability (good for default prediction)
y_pred:ndarray = (y_val_proba >= 0.3).astype(int)

list_model_metrics(y_val, y_pred, y_val_proba)

list_metrics_for_thresholds(y_val,  y_val_proba,[0.5, 0.4, 0.3, 0.2])