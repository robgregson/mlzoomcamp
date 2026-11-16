import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from constants import *

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight=None,
    random_state=42
)

df = pd.read_csv(prepared_data_file_path)

# print(df.columns)

# Separate features and target
X = df.drop(columns=['default'])
y = df['default']

# First split: 60% train, 40% remaining
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Second split: divide remaining 40% equally
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

decision_tree_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight=None,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_tree = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall:", recall_score(y_test, y_pred_tree))
print("F1:", f1_score(y_test, y_pred_tree))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

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

print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall:", recall_score(y_test, y_pred_tree))
print("F1:", f1_score(y_test, y_pred_tree))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))


y_prob = dt_model.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")


