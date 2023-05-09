import subprocess
import os

# Set log level to ERROR
os.environ["ALLENNLP_LOG_LEVEL"] = "ERROR"

# Define the dataset codes and pretrained models with corresponding output filenames
datasets_and_models = [
    ("correct_format_scierc", "scierc.tar.gz", "scierc.jsonl"),
    ("correct_format_genia", "genia.tar.gz", "genia.jsonl"),
    ("correct_format_ace05_relation", "ace05-relation.tar.gz", "ace_rels.jsonl"),
    ("correct_format_ace05_event", "ace05-event.tar.gz", "ace_event.jsonl"),
    ("correct_format_mechanic_coarse", "mechanic-coarse.tar.gz", "coarse.jsonl"),
    ("correct_format_mechanic_granular",
     "mechanic-granular.tar.gz", "granular.jsonl"),
]

# Iterate over the datasets_and_models and run the command for each one
for dataset_code, model_filename, output_filename in datasets_and_models:
    prepared_dataset = f"prepared_datasets/{dataset_code}"
    print(
        f"Processing {model_filename} with output file {output_filename} for dataset {dataset_code}...")
    subprocess.run([
        "allennlp", "predict", f"pretrained/{model_filename}", prepared_dataset,
        "--predictor", "dygie",
        "--include-package", "dygie",
        "--use-dataset-reader",
        "--output-file", f"predictions/correct_format_{output_filename}"
    ])
