from src.constants import *
from src.data_inspection import *
from src.data_preprocessing import *

__EXPERIMENTAL__ = False

##########################################################################
# load the data
###

# ignore the row which contains: "X1", X2, X3, ... X23, Y
df = pd.read_csv(data_file_path, skiprows=1)

df = normalise_column_names(df)

# show all columns - not just the first and last
list_columns(df)
list_column_details(df)
print(df.columns)

##########################################################################
# process
###

if __EXPERIMENTAL__:
    # when experimenting, just take the first few rows
    df = df.iloc[:25].copy()

df = encode_categorical(df, 'SEX', 'gender',
                        {
                            1: 'male',
                            2: None         # we only need one binary column
                            # no need for any more - the data set has no other values
                        }
                        )

df = encode_categorical(df, 'EDUCATION', 'educ',
                        {
                            0: None,  # value not defined
                            1: 'post_grad',
                            2: 'grad',
                            3: 'high',
                            4: None,  # ignore these - no idea what 'other' means
                            5: None,  # value not defined
                            6: None  # value not defined
                        }
                        )

df = encode_categorical(df, 'MARRIAGE', 'rel',
                        {
                            0: None,
                            1: 'married',
                            2: 'single',
                            3: None  # ignore 'other's
                        }
                        )

df = add_derived_behaviour_columns(df)
df = add_credit_utilisation_columns(df)         # should be ok - there are no missing values for LIMIT_BAL
df = add_trend_columns(df)
df = add_average_bill_columns(df).dropna(subset=['average_bill_amount'])
df = add_average_repayment_columns(df).dropna(subset=['average_payment_amount'])
df = add_repayment_proportion_columns(df).dropna(subset=['repayment_proportion_average_new'])


# remove the less relevant columns
df = df.drop(columns=['ID', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                      'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        )

##########################################################################
# display
###

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

##########################################################################
# data quality check & output
###

post_process_column_check(df)

spot_correlations(df)

answer = input("\nDo you want to continue to output the data? (y/n): ").strip().lower()

if answer in ["yes", "y"]:
    print("\nSaving data to output file: ", prepared_data_file_path)
    df.to_csv(prepared_data_file_path, index=False, header=True)
else:
    print("Stopping.")


