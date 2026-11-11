
from pathlib import Path
# import sys

import pandas as pd
# import numpy as np
# import pandas.api.types as pdt
# import seaborn as sns
# from matplotlib import pyplot as plt
# %matplotlib inline
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score

import data_inspection as inspection
from data_preprocessing import normalise_column_names, encode_categorical, add_derived_behaviour_columns

# load the data
folder_path = Path(__file__).parent
data_directory =  'data'
filename = 'credit_card_default_data.csv'
data_file_path = folder_path / data_directory / filename

# ignore the row which contains: "X1", X2, X3, ... X23, Y
df = pd.read_csv(data_file_path,skiprows=1)
# show all columns - not just the first and last

inspection.list_columns(df)
inspection.list_column_details(df)

print(df.columns)
# print(df[['ID', 'SEX', 'MARRIAGE', 'AGE']].head())

df = normalise_column_names(df)

df = encode_categorical(df, 'SEX', 'gender',
                        {
                            1: 'male',
                            2: 'female'
                            # no need for any more - the data set has no other values
                          }
                        )

df = encode_categorical(df, 'EDUCATION', 'educ',
                        {
                            0: None,    # value not defined
                            1: 'post_grad',
                            2: 'grad',
                            3: 'high',
                            4: None,    # ignore these - no idea what 'other' means
                            5: None,    # value not defined
                            6: None     # value not defined
                          }
                        )

df = encode_categorical(df, 'MARRIAGE', 'rel',
                        {
                            0: None,
                            1: 'married',
                            2: 'single',
                            3: None     # ignore 'other's
                          }
                        )

df = add_derived_behaviour_columns(df)

print("\n---\n")
pd.set_option('display.width', 240)
pd.set_option('display.max_columns', None)
# print(df.columns)
# print(df[['ID', 'AGE', 'marital_married', 'marital_single', 'marital_other', 'marital_unknown', 'gender_male', 'gender_female']].head())
# print(df.head(10))
# show rows where latest payment is 8 months late
# print(df[df['PAY_1'] == 8].head(10))
# print(df[df['PAY_1'] == -2].head(10))
# print(df[df['PAY_1'] == -1].head(10))
print(df[df['PAY_1'] == 0].filter(regex='^(?!.*AMT)').head(10))


