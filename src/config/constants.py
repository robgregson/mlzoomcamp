from src.config.paths import DATA_DIRECTORY

data_directory = 'data'
filename = 'credit_card_default_data.csv'
output_filename = 'credit_card_processed_data.csv'
data_file_path = DATA_DIRECTORY / filename
prepared_data_file_path = DATA_DIRECTORY / output_filename

data_year:int = 2005
annual_taiwan_inflation:float = 0.0136
currency_fields = ['LIMIT_BAL','PAY_1','BILL_AMT1','PAY_AMT1']
fx_curr_to_twd = {
    'TWD' : 1,
    'USD' : 33,
    'GBP' : 46,
    'EUR': 37.5
}