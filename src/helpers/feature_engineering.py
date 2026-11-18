import pandas

from src.helpers.data_preprocessing import *


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = normalise_column_names(df)

    df = encode_categorical(df, 'SEX', 'gender',
                            {
                                1: 'male',
                                2: None  # we only need one binary column
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
    df = add_credit_utilisation_columns(df)  # should be fine - there are no missing values for LIMIT_BAL
    df = add_trend_columns(df)
    df = add_average_bill_columns(df).dropna(subset=['average_bill_amount'])
    df = add_average_repayment_columns(df).dropna(subset=['average_payment_amount'])
    df = add_repayment_proportion_columns(df).dropna(subset=['repayment_proportion_average_new'])

    # remove the less relevant columns
    df = df.drop(columns=['ID', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                          'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                          'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                 )

    return df
