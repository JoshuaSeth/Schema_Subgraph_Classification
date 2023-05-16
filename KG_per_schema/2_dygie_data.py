'''Creates dygie compatible datasets from the default data'''

import subprocess
import glob
import os

# Some variables for the operation
python_executive = "python3.8"
default_sent_folder = 'KG_per_schema/data/sents/'
dygie_formatter_path = "streamlit_compare_schemas/ORKG_parsers/dygiepp/scripts/new-dataset/format_new_dataset.py"

# Define the dataset codes (needed for dygie to correctly understand dataset)
dataset_codes = [
    "scierc",
    "scierc",
    "genia",
    "genia",
    "chemprot",
    "ace05",
    "ace-event",
    "None",
    "covid-event",
]

for origin_path in glob.glob(f"{default_sent_folder}*.txt"):

    # Iterate over the dataset_codes and run the command for each one
    for dataset_code in dataset_codes:
        target_filename = os.path.basename(origin_path) + '_' + dataset_code
        target_path = f"KG_per_schema/data/dygie_data/{target_filename}"

        print(f"Processing dataset code {dataset_code}...")

        subprocess.run([python_executive, dygie_formatter_path,
                       origin_path, target_path, dataset_code, "--use-scispacy"])
