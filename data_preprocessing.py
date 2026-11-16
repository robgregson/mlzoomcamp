import pandas as pd
import numpy as np

def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df:
    :return: the dataframe with normalised columns
    """
    # rename the default column
    df = df.rename(columns={'default payment next month': 'default'})
    # rename the PAY_0 column to standardise
    df = df.rename(columns={'PAY_0': 'PAY_1'})
    return df

def encode_categorical(df: pd.DataFrame, src_column: str, new_cols_prefix: str, encoding_dict: dict[int, str | None]) -> pd.DataFrame:
    """
    :param df:
    :param src_column: the column to encode
    :param new_cols_prefix: the new colum
    ns prefix
    :param encoding_dict: the encoding dictionary - what the original column values mean (or None - don't encode)
    :return:
    """
    # convert categorical variable into dummy variables
    dummies = pd.get_dummies(df[src_column], prefix=new_cols_prefix, dtype=int)

    # remove the original column
    df = df.drop(columns=[src_column])
    # join up the new columns
    df = df.join(dummies)
    # dataframe[dummies.columns] = dummies

    # rename the columns we want to keep (non-None)
    rename_dict = {
        f'{new_cols_prefix}_{key}': f'{new_cols_prefix}_{value}'
        for key, value in encoding_dict.items()
        if value is not None
    }
    df = df.rename(columns=rename_dict)

    # drop the 'None' columns
    drop_list = [
        f'{new_cols_prefix}_{key}'
        for key, value in encoding_dict.items()
        if value is None
    ]
    df = df.drop(columns=drop_list, errors='ignore')

    # don't worry about wasting memory in re-assignment - modern Python works it out
    return df

def add_derived_behaviour_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df:
    :return: the dataframe with derived columns added
    """
    pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    total_months = len(pay_cols)

    # proportion of months in which cardholder adds nothing to the credit balance
    df['no_spend_months_proportion'] = df[pay_cols].eq(-2).sum(axis=1) / total_months
    # proportion of months in which cardholder pays off the outstanding balance
    df['payoff_months_proportion'] = df[pay_cols].eq(-1).sum(axis=1) / total_months
    # proportion of months in which cardholder rolls the outstanding balance over to the next month
    df['rollover_months_proportion'] = df[pay_cols].eq(0).sum(axis=1) / total_months

    # Average late payment severity, normalised against the maximum delay of 8 - default to 0
    late_payment_sum = df[pay_cols].where(df[pay_cols] > 0, 0).sum(axis=1)
    late_payment_months = df[pay_cols].gt(0).sum(axis=1)
    df['late_payment_rating'] = (late_payment_sum / (late_payment_months * 8)).fillna(0)

    return df

def add_credit_utilisation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df:
    :return: the dataframe with derived credit utilisation columns added
    """

    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

    non_zero_limit = df['LIMIT_BAL'] != 0

    # Current proportion of available credit being used
    # df['latest_credit_utilisation'] = (df['BILL_AMT1'] / df['LIMIT_BAL'].where(non_zero_limit))
    # Average proportion of available credit being used over six months
    df['average_credit_utilisation'] = (df[bill_cols].mean(axis=1) /df['LIMIT_BAL'].where(non_zero_limit))
    # Highest proportion of available credit being used over six months
    # df['max_credit_utilisation'] = (df[bill_cols].max(axis=1) /df['LIMIT_BAL'].where(non_zero_limit))

    return df

def add_trend_columns(df: pd.DataFrame) -> pd.DataFrame:

    # the data is organised in a weird way - AMT_1 is current, AMT_2 is previous month, so
    # +ve slope meaning borrowing more, -ve means paying off
    # so when you inspect the data in a table format, the trend seems to go the other way
    bill_amount_cols = [f'BILL_AMT{i}' for i in [1, 2, 3, 4, 5, 6]]
    month = np.array([0, -1, -2, -3, -4, -5])

    def calculate_bill_trend(row: pd.Series) -> pd.Series:
        slope, intercept = np.polyfit(month, row[bill_amount_cols].values, 1)
        # a Series is like a dict and suitable for this situation
        # don't use bill_trend_intercept after correlation investigation
        # return pd.Series({'bill_trend_slope': slope,
        #                   'bill_trend_intercept': intercept})
        return pd.Series(slope)

    df[['bill_trend_slope']] = df.apply(calculate_bill_trend, axis=1)

    return df

def add_average_bill_columns(df: pd.DataFrame) -> pd.DataFrame:

    bill_amt_cols = [f'BILL_AMT{i}' for i in [1, 2, 3, 4, 5, 6]]

    df['average_bill_amount'] = df[bill_amt_cols].mean(axis=1)

    return df

def add_average_repayment_columns(df: pd.DataFrame) -> pd.DataFrame:

    pay_amt_cols = [f'PAY_AMT{i}' for i in [1, 2, 3, 4, 5, 6]]

    df['average_payment_amount'] = df[pay_amt_cols].mean(axis=1)

    return df

def add_repayment_proportion_columns(df: pd.DataFrame) -> pd.DataFrame:

    repayment_proportions: list[pd.Series] = []

    for i in [1, 2, 3, 4, 5]:
        # Remember this is all Series stuff, not individual rows
        bill_amounts = df[f'BILL_AMT{i + 1}']
        valid_bills = bill_amounts.gt(0)
        # this stores 5 figures per row
        repayment_proportions.append(
            (df[f'PAY_AMT{i}'] / bill_amounts.where(valid_bills))
        )

    df['repayment_proportion_average_new'] = pd.concat(repayment_proportions, axis=1).mean(axis=1)

    return df
