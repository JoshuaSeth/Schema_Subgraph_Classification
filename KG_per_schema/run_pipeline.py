'''Run the full pipeline: 1. data loading, 2. dygie data preparing, 3. running predictions and creating prediction datasets'''

# from data import create_sentence_datasets
from dygie_data import convert_dygie_compatible_datasets
from predict import create_prediction_datasets

# Run every step of the pipeline. Each step uses the data generated by the previous part. These can also be run separately by running their respective files.

# 1.
# create_sentence_datasets()

# 2.
# convert_dygie_compatible_datasets()

# # 3.
create_prediction_datasets(schemas=[
    'scierc',
    'None', 'genia', 'covid-event', 'ace05', 'ace-event'], use_cached=False)
