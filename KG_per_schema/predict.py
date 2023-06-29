'''File that does the NER and RE on the actual sentences. Does predictions with and without context, with AND or OR research challenges and directions, and for different models. 

Currently schema is tied to model, so the set up is not (model x schema) but certain schema's are only available for certain models. Hope to extend this in the future to: (model x schema x context x OR/AND)

In the further future this should be extended to (model x schema x context x OR/AND x MODELLING: (n-ary, triples) x OUTPUT FORMAT: (query, multi triples))'''


import subprocess
import os
from utils import get_model_fname, project_path
import glob
from tqdm import tqdm
import json


# Little less verbose logs
os.environ["ALLENNLP_LOG_LEVEL"] = "ERROR"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Variables
dygie_data_dir_path = project_path + '/KG_per_schema/data/dygie_data/'
base_data_dir_path = project_path + '/KG_per_schema/data/sents/'
output_dir_path = project_path + '/KG_per_schema/data/predictions/'
dygie_dir_path = project_path + \
    "/old/streamlit_compare_schemas/ORKG_parsers/dygiepp/"
interpreter = '/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/.37venv/bin/python'


def create_prediction_datasets(schemas=['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event', 'spacy'], use_cached=True):
    '''
    Use the dygie schema models to do predictions for all datasets.

    Parameters
    ------------
        schemas: list
            Which schemas to use for prediction. Dyfie datafiles for schemas not in the list will be ignored. Default is all available schemas. A list of [scierc, None (= mechanic coarse), genia, covid-event (= mechanic granular), ace05, ace-event]
        use_cached: bool
            Whether to use the cached predictions (if available) or to create new ones. Default: False

    Return
    -----------
        None
            Saves the predictions in the data/predictions folder.'''
    # Iterate over the datasets_and_models and run the command for each one
    for dygie_data_fpath in tqdm(glob.glob(f"{dygie_data_dir_path}*")):
        # Check if it is not an malformed file

        properties = os.path.basename(dygie_data_fpath).split('_')
        if not all(char.isdigit() for char in properties[-2]):

            # Filenames
            fname = os.path.basename(dygie_data_fpath)
            schema = fname.split('_')[0]
            model_fpath = dygie_dir_path + \
                'pretrained/' + get_model_fname(fname)
            output_fpath = output_dir_path + fname

            # Only perform prediction if desired schema and not caching or no cached file exists
            if schema in schemas and (not os.path.isfile(output_fpath) or not use_cached):
                print(
                    f"Processing {fname} with {get_model_fname(fname)}, for {os.path.basename(output_fpath)}...")

                # Without specifying the python version this uses python3.8 for me and that is the only way I can get it working (together with the dygie requirements.txt)
                cmd = f'''"{interpreter}" -m allennlp predict "{model_fpath}" "{dygie_data_fpath}"  --include-package dygie   --predictor dygie  --use-dataset-reader  --cuda-device "-1" --output-file "{output_fpath}"'''
                subprocess.run(cmd, shell=True, cwd=dygie_dir_path)
        else:
            print(f"Skipping {dygie_data_fpath} because it is malformed")
    # For non-dygie (spacy) predictions we simply read from the base data but write the data in the same format
    if 'spacy' in schemas:
        import spacy
        # import en_core_sci_scibert
        # import en_core_web_lg
        # import en_core_web_sm

        for model in ['en_core_web_sm', 'en_core_web_lg', 'en_core_sci_scibert']:
            nlp = spacy.load(model)

            for base_data_fpath in tqdm(glob.glob(f"{base_data_dir_path}*")):
                mode = os.path.basename(base_data_fpath).split('_')[1]
                prediction = {"doc_key": os.path.basename(
                    base_data_fpath), "dataset": "spacy"}

                with open(base_data_fpath, 'r') as f:
                    sents = f.read()

                print(base_data_fpath)
                doc = nlp(sents)

                prediction['sentences'] = [[token.text for token in sent]
                                           for sent in doc.sents]

                predicted_ners = []
                for sent in doc.sents:
                    new_sent = []
                    ent_names = {ent.text: ent.label_ for ent in sent.ents}

                    for token in sent:
                        if token.text in ent_names:
                            new_sent.append(
                                (token.text, ent_names[token.text]))
                        else:
                            new_sent.append(token.text)

                    predicted_ners.append(new_sent)

                prediction['tagged_sents'] = predicted_ners

                output_fname = 'spacy-' + \
                    model.replace(
                        '_', '') + '_' + os.path.basename(base_data_fpath).replace('.txt', '') + '_0'
                with open(output_dir_path + output_fname, 'w') as f:
                    json.dump(prediction, f)


if __name__ == '__main__':
    create_prediction_datasets(schemas=['spacy'])
