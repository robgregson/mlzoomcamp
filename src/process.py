import pandas as pd
from src.config.constants import *
from src.helpers.data_inspection import *

__EXPERIMENTAL__ = False

from src.helpers.feature_engineering import engineer_features

# ---------------------------------------------------------------------
# load the data

# ignore the row which contains: "X1", X2, X3, ... X23, Y
df = pd.read_csv(DATA_FILEy_PATH, skiprows=1)

if __EXPERIMENTAL__:
    # when experimenting, just take the first few rows
    df = df.iloc[:25].copy()


# show all columns - not just the first and last
list_columns(df)
list_column_details(df)
print(df.columns)

# ---------------------------------------------------------------------
# process


df = engineer_features(df)

# ---------------------------------------------------------------------
# display

print("\n--x--\n")
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', None)
print(df.columns)

print(df.head(25))
# show rows where latest payment is 8 months late
# print(df[df['PAY_1'] == 8].head(10))
# print(df[df['PAY_1'] == -2].head(10))
# print(df[df['PAY_1'] == -1].head(10))
# exclude AMT rows
# print(df[df['PAY_1'] == 0].filter(regex='^(?!.*AMT)').head(10))

cols = df.columns[df.isna().any()].tolist()
if len(cols) == 0:
    print("\nAll columns contain numeric data")
else:
    print("\ncolumns containing nan data:")
    print(df.columns[df.isna().any()].tolist())

# ---------------------------------------------------------------------
# data quality check & output

post_process_column_check(df)

spot_correlations(df)

answer = input("\nDo you want to continue to output the data? (y/n): ").strip().lower()

if answer in ["yes", "y"]:
    print("\nSaving data to output file: ", PREPARED_DATA_FILE_PATH)
    df.to_csv(PREPARED_DATA_FILE_PATH, index=False, header=True)
else:
    print("Stopping.")


