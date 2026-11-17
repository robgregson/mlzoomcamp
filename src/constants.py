from pathlib import Path

folder_path = Path(__file__).parent.parent

# print(folder_path)

data_directory = 'data'
filename = 'credit_card_default_data.csv'
output_filename = 'credit_card_processed_data.csv'
data_file_path = folder_path / data_directory / filename
prepared_data_file_path = folder_path / data_directory / output_filename