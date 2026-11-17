import pandas as pd
import numpy as np
import pandas.api.types as pdt
import seaborn as sns
import matplotlib.pyplot as plt


def list_columns(df: pd.DataFrame):
  numeric_columns = list(df.select_dtypes(include=['number']).columns)
  print("Numeric columns:")
  print(numeric_columns)

  text_columns = list(df.select_dtypes(include=['object', 'category']).columns)
  print("Text columns:")
  print(text_columns)
  print("\n---\n")

def list_column_details(df: pd.DataFrame):
  for col in df.columns:
    print(f"Column: '{col}'")
    if pdt.is_numeric_dtype(df[col]):
      print("NUMERIC. \t\tMin:",df[col].min(), '\tmax:',df[col].max())
    else:
      print("TEXT. Example values:\t", df[col].unique()[:5])

    # noinspection bad-assignment
    unique_values_count:int = df[col].nunique()
    print('Unique values:',unique_values_count)
    if unique_values_count< 10:
      print(df[col].value_counts().sort_index())

    if df[col].isnull().any() or df[col].isna().any():
      print('Missing values:',df[col].isnull().sum())
    else:
      print('No missing values')

    print()

def post_process_column_check(df: pd.DataFrame ):

  # Check for NaN values
  nan_counts: pd.Series = df.isna().sum()
  nan_columns = nan_counts[nan_counts.gt(0)]

  if nan_columns.empty:
    print("\nNo NaN values found")
  else:
    print("\nColumns containing NaN values:")
    print(nan_columns)

  # Check for infinite values
  numeric_df = df.select_dtypes(include=np.number)
  inf_counts: pd.Series = np.isinf(numeric_df).sum()
  inf_columns = inf_counts[inf_counts.gt(0)]

  if inf_columns.empty:
    print("\nNo infinite values found")
  else:
    print("\nColumns containing infinite values:")
    print(inf_columns)

  # Overall summary
  print(f"\nTotal rows: {len(df)}")
  print(f"Rows containing NaN: {df.isna().any(axis=1).sum()}")
  print(f"Rows containing infinity: {np.isinf(numeric_df).any(axis=1).sum()}")

def spot_correlations(df: pd.DataFrame):
  # Select numerical features

  feature_columns = df.columns.tolist()
  columns_to_remove = ['LIMIT_BAL', 'AGE', 'PAY_1', 'BILL_AMT1', 'PAY_AMT1', 'default',
   'gender_male', 'educ_post_grad', 'educ_grad', 'educ_high', 'rel_married', 'rel_single']
  for col in columns_to_remove:
    if col in feature_columns:
      feature_columns.remove(col)

  corr = df[feature_columns].corr(method='spearman')

  plt.figure(figsize=(14, 10))

  sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f'
  )

  plt.title('Feature Correlation Matrix')
  plt.tight_layout()
  plt.show()