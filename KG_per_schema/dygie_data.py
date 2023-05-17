'''Creates dygie compatible datasets from the default data'''

import subprocess
import glob
import os
from utils import project_path
import json

# Some variables for the operation
python_kernel = "python3.11"
sents_dir_path = project_path + '/KG_per_schema/data/sents/'
dygie_data_dir_path = project_path + '/KG_per_schema/data/dygie_data/'
dygie_formatter_fpath = project_path + \
    "/streamlit_compare_schemas/ORKG_parsers/dygiepp/scripts/new-dataset/format_new_dataset.py"


print('project_path:', project_path)
print('default_sent_folder:', sents_dir_path)
print('dygie_formatter_path:', dygie_formatter_fpath)

# Define the dataset codes (needed for dygie to correctly understand dataset)
dataset_codes = [
    "scierc",
    "genia",
    # "chemprot",
    "ace05",
    "ace-event",
    "None",
    "covid-event",
]


def convert_dygie_compatible_datasets():
    # Iterate over the dataset_codes and run the command for each one
    for sents_fpath in glob.glob(f"{sents_dir_path}*.txt"):
        for dataset_code in dataset_codes:
            target_fname = (
                dataset_code + '_' + os.path.basename(sents_fpath)).replace('.txt', '')
            target_fpath = dygie_data_dir_path + target_fname

            print(
                f"Processing dataset code {dataset_code} for target {target_fname}...")

            # Call the dygie dataset converter script
            subprocess.run([python_kernel, dygie_formatter_fpath,
                            sents_fpath, target_fpath, dataset_code, "--use-scispacy"])

    # Postprocessing needs to happen since spacy short sentences crash dygie
    for dygie_data_fpath in glob.glob(f"{dygie_data_dir_path}"):
        # Filter the sentences that are too short
        with open(dygie_data_fpath, 'r') as f:
            data = json.load(f)
        sents = [sent for sent in data['sentences'] if len(sent) > 3]
        data['sentences'] = sents
        with open(dygie_data_fpath, 'w') as f:
            json.dump(data, f)  # Overwrite the file


if __name__ == '__main__':
    convert_dygie_compatible_datasets()
