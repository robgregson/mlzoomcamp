import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report, roc_auc_score
)
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split

from src.config.paths import DATA_DIRECTORY


def choose_threshold(y_true, y_prob):
    """Choose the probability threshold that maximises F1."""
    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Train and save a Random Forest default prediction model."
    )
    parser.add_argument(
        "--data",
        default=str(DATA_DIRECTORY / "credit_card_processed_data.csv"),
        help="Path to the processed CSV dataset.",
    )
    parser.add_argument(
        "--target",
        default="default",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--output",
        default="train.pkl",
        help="Path for the saved pickle model bundle.",
    )
    args = parser.parse_args()

    # 1. Load the processed dataset
    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' was not found in the dataset.")

    if df[args.target].isna().any():
        raise ValueError("The target column contains missing values.")

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    # 2. Split into 60% training, 20% validation and 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    # 3. Create and train the Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 4. Select threshold using validation data only
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold, validation_f1 = choose_threshold(y_val, val_probabilities)

    print(f"Selected threshold: {threshold:.2f}")
    print(f"Validation F1:      {validation_f1:.4f}")

    val_predictions = (val_probabilities >= threshold).astype(int)
    print("\nValidation metrics:")
    print(f"Accuracy:  {accuracy_score(y_val, val_predictions):.4f}")
    print(f"Precision: {precision_score(y_val, val_predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_val, val_predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_val, val_predictions, zero_division=0):.4f}")

    # 5. Final evaluation on the held-out test data
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    print("\nTest metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, test_predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, test_probabilities):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))

    # 6. Save model and prediction metadata together
    model_bundle = {
        "model": model,
        "threshold": threshold,
        "feature_columns": X.columns.tolist(),
        "target": args.target,
        "model_type": "RandomForestClassifier",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(model_bundle, file)

    print(f"\nSaved model bundle to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
