import datetime
from src.config.constants import (
    data_year,
    annual_taiwan_inflation,
    currency_fields,
    fx_curr_to_twd
)
from src.helpers.data_preprocessing import *
from src.helpers.feature_engineering import engineer_features


def deflate_currency_values(df: pd.DataFrame)  -> pd.DataFrame:

    year_diff = datetime.datetime.now().year - data_year
    inflation_factor = annual_taiwan_inflation ** year_diff

    for f in currency_fields:
        df[f] = df[f] / inflation_factor

    return df

def convert_currency_values(df: pd.DataFrame, currency_3_letters) -> pd.DataFrame:

    fx_conversion_rate = fx_curr_to_twd.get(currency_3_letters, 1.0)

    for f in currency_fields:
        df[f] = df[f] * fx_conversion_rate

    return df


def preprocess_input(input_data: dict, currency_3_letters: str= 'TWD') -> pd.DataFrame:

    # convert json dict into df
    df = pd.DataFrame([input_data])
    # convert currency to 2005 Taiwan Dollars
    df = convert_currency_values(df, currency_3_letters)
    df = deflate_currency_values(df)
    # apply the same feature engineering as the training dataset
    df = engineer_features(df)

    return df
