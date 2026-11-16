# Credit Card Default Prediction

## Project Overview

This project investigates the use of machine learning to predict whether a credit card customer is likely to default on their payment obligations.

Using historical customer, credit limit, repayment and billing information, the project develops and evaluates classification models to identify customers who may default.

The primary objective is to build a predictive model that can distinguish between customers who default and those who do not, while considering the challenges associated with an imbalanced dataset.

## Dataset

The dataset contains customer demographic information and historical credit card payment and billing records.

The target variable is:

- `default`: Indicates whether a customer defaulted (1) or did not default (0).

The dataset contains a range of numerical and categorical features, including credit limits, age, repayment status, bill amounts and payment amounts.

## Methodology

The project follows a structured machine learning workflow.

### 1. Data Preparation

Initial data preparation includes:

- Inspecting the dataset and its structure.
- Checking for missing values and duplicate records.
- Examining data types and feature distributions.
- Separating the target variable from the predictor features.
- Splitting the data into training and testing sets.

### 2. Feature Engineering

Additional features are created to capture patterns in customer behaviour that may help predict default.

These include measures of credit utilisation, average billing amounts and trends in billing and repayment behaviour.

Feature engineering aims to provide the models with more meaningful representations of customer financial behaviour.

### 3. Feature Selection

Spearman correlation analysis is used to investigate relationships between engineered features and identify potential redundancy.

The following features were removed because of strong correlations with other features:

| Removed feature | Reason |
|---|---|
| `latest_credit_utilisationn` | Strongly correlated with `average_credit_utilisation` |
| `max_credit_utilisation` | Strongly correlated with `average_credit_utilisation` |
| `bill_trend_intercept` | Strongly correlated with `average_bill_amount` |

This reduces unnecessary feature redundancy and helps simplify the modelling process.

### 4. Model Development

Three classification approaches are investigated:

**Logistic Regression**

Used as a baseline classification model. Features are standardised using `StandardScaler` before training.

**Decision Tree**

A tree-based classification model that identifies patterns in the data through a sequence of feature-based decisions.

Hyperparameter optimisation is performed to investigate the effects of tree depth, minimum samples per split, minimum samples per leaf, splitting criteria and class weighting.

**Random Forest**

An ensemble learning approach that combines predictions from multiple decision trees. It is investigated to determine whether combining several trees improves predictive performance and generalisation.

### 5. Model Evaluation

Models are evaluated using the following classification metrics:

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** The proportion of predicted defaults that are actual defaults.
- **Recall:** The proportion of actual defaults correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across different classification thresholds.

The confusion matrix is also used to examine true positives, true negatives, false positives and false negatives.

Particular attention is given to recall and F1-score for the default class, as identifying customers who may default is an important objective.

Classification threshold adjustments are also investigated to explore the trade-off between precision and recall.

## Model Optimisation

Decision tree hyperparameters are investigated using `RandomizedSearchCV` with stratified five-fold cross-validation.

The search evaluates multiple combinations of hyperparameters using F1-score as the optimisation metric.

This helps identify a model configuration that performs consistently across different training and validation folds.

## Technologies and Libraries

The project is implemented in Python using:

- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical operations.
- **Matplotlib / Seaborn:** Data visualisation.
- **Scikit-learn:** Data preprocessing, model development, hyperparameter optimisation and evaluation.
- **Jupyter Notebook:** Exploratory analysis and experimentation.

## Running the Project

1. Clone or download this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Run the notebook cells sequentially to reproduce the data preparation, feature engineering, model training and evaluation.

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Conclusion

This project explores how machine learning can be applied to credit card default prediction.

By comparing Logistic Regression, Decision Trees and Random Forests, the project investigates the strengths and limitations of different classification approaches.

The evaluation considers not only overall predictive accuracy but also the ability to identify customers who actually default. This is particularly important when working with imbalanced classification problems, where a model can achieve relatively high accuracy while failing to identify a substantial proportion of the minority class.

The final model selection is based on a comparison of predictive performance, with particular consideration given to precision, recall and F1-score.

## Future Improvements

Potential areas for further development include:

- More extensive feature importance and model interpretation analysis.
- Further hyperparameter optimisation.
- Exploring alternative ensemble methods such as Gradient Boosting.
- Evaluating model calibration and decision thresholds.
- Investigating the business implications of false positive and false negative predictions.
