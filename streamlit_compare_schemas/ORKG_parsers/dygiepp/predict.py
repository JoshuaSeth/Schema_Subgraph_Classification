'''File that does the NER and RE on the actual sentences. Does predictions with and without context, with AND or OR research challenges and directions, and for different models. 

Currently schema is tied to model, so the set up is not (model x schema) but certain schema's are only available for certain models. Hope to extend this in the future to: (model x schema x context x OR/AND)

In the further future this should be extended to (model x schema x context x OR/AND x MODELLING: (n-ary, triples) x OUTPUT FORMAT: (query, multi triples))'''


import subprocess
import os
# from utils import get_model_fname, project_path
import glob
from os.path import dirname


project_path: str = '/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/'


def get_model_fname(filename):
    '''Get the appropriate model name from the training file'''
    filename = filename.split('_')[0]
    if filename == "scierc":
        return "scierc.tar.gz"
    if filename == "genia":
        return "genia.tar.gz"
    if filename == "ace05":
        return "ace05-relation.tar.gz"
    if filename == "ace-event":
        return "ace05-event.tar.gz"
    if filename == "None":
        return "mechanic-granular.tar.gz"
    if filename == "covid-event":
        return "mechanic-coarse.tar.gz"


# Little less verbose logs
os.environ["ALLENNLP_LOG_LEVEL"] = "WARNING"

# Variables
dygie_data_dir_path = project_path + '/KG_per_schema/data/dygie_data/'
output_dir_path = project_path + '/KG_per_schema/data/predictions/'
dygie_dir_path = project_path + "/streamlit_compare_schemas/ORKG_parsers/dygiepp/"


def create_prediction_datasets():
    # Iterate over the datasets_and_models and run the command for each one
    for dygie_data_fpath in glob.glob(f"{dygie_data_dir_path}*.jsonl"):

        # Filenames
        fname = os.path.basename(dygie_data_fpath)
        model_fpath = dygie_dir_path + 'pretrained/' + get_model_fname(fname)
        output_fpath = output_dir_path + fname.replace('.jsonl', '')

        print(f"Processing {fname}...")

    #     subprocess.run([
    #     "allennlp", "predict", f"pretrained/{model_filename}", prepared_dataset,
    #     "--include-package", "dygie",
    #     "--predictor", "dygie",

    #     "--use-dataset-reader",
    #     "--output-file", f"predictions/context_{output_filename}"
    # ])

        cmd = f' allennlp predict "{model_fpath}" "{dygie_data_fpath}" --include-package dygie  --predictor dygie --use-dataset-reader --output-file "{output_fpath}"'
        subprocess.run(cmd, shell=True, cwd=dygie_dir_path)


if __name__ == '__main__':
    create_prediction_datasets()
