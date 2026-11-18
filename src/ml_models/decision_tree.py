import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from src.helpers.data_modelling import load_training_data, list_model_metrics, list_metrics_for_thresholds

X_train, y_train, X_val, y_val, X_test, y_test = load_training_data()

# train the model
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight=None,
    random_state=42
)
dt_model.fit(X_train, y_train)

y_pred_tree = dt_model.predict(X_test)

list_model_metrics(y_val, y_pred_tree)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

print("\n\ndepth experiment")

results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=depth,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        'depth': depth,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

print(results_df.round(4).to_string(index=False))

print("\n\n leaf size experiment")

results = []

for leaf_size in [5, 10, 20, 50, 100]:

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=leaf_size,
        class_weight=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        'min_samples_leaf': leaf_size,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    })

print(pd.DataFrame(results).round(4).to_string(index=False))


print("\n\n split experiment")

results = []

for split_size in [10, 20, 50, 100, 200]:

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=split_size,
        min_samples_leaf=100,
        class_weight=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        'min_samples_split': split_size,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    })

print(pd.DataFrame(results).round(4).to_string(index=False))

print("\n\n weight experiment")


results = []

for weight in [None, 'balanced']:

    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=100,
        class_weight=weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results.append({
        'class_weight': weight,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    })

print(pd.DataFrame(results).round(4).to_string(index=False))


print("\n\nbest model")

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=100,
    class_weight='balanced',
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_tree = dt_model.predict(X_test)
y_prob = dt_model.predict_proba(X_test)[:, 1]

list_model_metrics(y_test, y_pred_tree, y_prob)

list_metrics_for_thresholds(y_test, y_pred_tree,[0.3, 0.4, 0.5, 0.6, 0.7])
