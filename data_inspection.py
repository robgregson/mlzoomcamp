# import pandas as pd
# import numpy as np
import pandas.api.types as pdt

def list_columns(df):
  numeric_columns = list(df.select_dtypes(include=['number']).columns)
  print("Numeric columns:")
  print(numeric_columns)

  text_columns = list(df.select_dtypes(include=['object', 'category']).columns)
  print("Text columns:")
  print(text_columns)
  print("\n---\n")

def list_column_details(df):
  for col in df.columns:
    print(f"Column: '{col}'")
    if pdt.is_numeric_dtype(df[col]):
      print("NUMERIC. \t\tMin:",df[col].min(), '\tmax:',df[col].max())
    else:
      print("TEXT. Example values:\t", df[col].unique()[:5])

    unique_values_count = df[col].nunique()
    print('Unique values:',unique_values_count)
    if unique_values_count < 10:
      print(df[col].value_counts().sort_index())

    if df[col].isnull().any() or df[col].isna().any():
      print('Missing values:',df[col].isnull().sum())
    else:
      print('No missing values')

    print()