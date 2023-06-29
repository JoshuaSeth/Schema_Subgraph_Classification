'''Creates dygie compatible datasets from the default data'''

import subprocess
import glob
import os
from utils import project_path
import json
from tqdm import tqdm
from copy import deepcopy

# Some variables for the operation
python_kernel = ".venv/bin/python"
sents_dir_path = project_path + '/KG_per_schema/data/sents/'
dygie_data_dir_path = project_path + '/KG_per_schema/data/dygie_data/'
dygie_formatter_fpath = project_path + \
    "/old/streamlit_compare_schemas/ORKG_parsers/dygiepp/scripts/new-dataset/format_new_dataset.py"


print('project_path:', project_path)
print('default_sent_folder:', sents_dir_path)
print('dygie_formatter_path:', dygie_formatter_fpath)

# Define the dataset codes (needed for dygie to correctly understand dataset)
dataset_codes = [
    # "scierc",
    "genia",
    # "chemprot",
    "ace05",
    "ace-event",
    "None",
    "covid-event",
]


def convert_dygie_compatible_datasets(convert=True, postprocess=True, subset=True):
    # Iterate over the dataset_codes and run the command for each one
    if convert:
        print('Converting datasets to dygie compatible format...')
        for sents_fpath in tqdm(glob.glob(f"{sents_dir_path}*.txt")):
            for dataset_code in tqdm(dataset_codes, leave=False):

                target_fname = (
                    dataset_code + '_' + os.path.basename(sents_fpath)).replace('.txt', '')
                target_fpath = dygie_data_dir_path + target_fname

                print(target_fpath)

                # Call the dygie dataset converter script
                subprocess.run([python_kernel, dygie_formatter_fpath,
                                sents_fpath, target_fpath, dataset_code, "--use-scispacy"])

    if postprocess:
        # Postprocessing needs to happen since spacy short sentences crash dygie
        print('Postprocessing dygie compatible datasets...')
        for dygie_data_fpath in tqdm(glob.glob(f"{dygie_data_dir_path}*")):
            try:
                print('dygie_data_fpath')
                # Filter the sentences that are too short
                with open(dygie_data_fpath, 'r') as f:
                    data = json.load(f)
                sents = [sent for sent in data['sentences'] if len(sent) > 3]
                data['sentences'] = sents
                with open(dygie_data_fpath, 'w') as f:
                    json.dump(data, f)  # Overwrite the file
            except Exception as e:
                print(e)
                print('Malformed file:', dygie_data_fpath)

    if subset:
        # We need to take subsets since dygie silently crashes on too large datasets
        print('Postprocessing dygie compatible datasets...')
        for dygie_data_fpath in tqdm(glob.glob(f"{dygie_data_dir_path}*")):
            with open(dygie_data_fpath, 'r') as f:
                data = json.load(f)
            n = 0
            pbar = tqdm(total=len(data['sentences']))

            while n < len(data['sentences']):
                # Take a subset of 40 sentences
                subset_data = deepcopy(data)
                subset = data['sentences'][n:n+40]
                subset_data['sentences'] = subset
                with open(dygie_data_fpath + '_' + str(n), 'w') as f:
                    json.dump(subset_data, f)
                n += 40
                pbar.update(40)
            # Remove the old large file
            os.remove(dygie_data_fpath)
            pbar.close()


if __name__ == '__main__':
    convert_dygie_compatible_datasets()
