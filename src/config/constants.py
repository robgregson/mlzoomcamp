from src.config.paths import DATA_DIRECTORY

DATA_FILE_PATH = DATA_DIRECTORY / 'credit_card_default_data.csv'
PREPARED_DATA_FILE_PATH = DATA_DIRECTORY / 'credit_card_processed_data.csv'

BASELINE_YEAR:int = 2005
ANNUAL_TAIWAN_INFLATION:float = 0.0136
CURRENCY_FIELDS = ['LIMIT_BAL', 'PAY_1', 'BILL_AMT1', 'PAY_AMT1']
FX_CURR_TO_TWD = {
    'TWD' : 1,
    'USD' : 33,
    'GBP' : 46,
    'EUR': 37.5
}

PICKLE_FILENAME = "train.pkl"