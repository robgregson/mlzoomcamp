import pandas as pd
# import numpy as np
# import pandas.api.types as pdt

def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df:
    :return: the dataframe with normalised columns
    """
    # rename the default column
    df = df.rename(columns={'default payment next month': 'default_on_payment_next_month'})
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
    df = df.drop(columns=drop_list)

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