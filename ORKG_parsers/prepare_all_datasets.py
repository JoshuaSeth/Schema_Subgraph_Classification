import subprocess

# Define the dataset codes
dataset_codes = {
    "scierc": "scierc",
    "scierc_lightweight": "scierc",
    "genia": "genia",
    "genia_lightweight": "genia",
    "chemprot": "chemprot",
    "ace05_relation": "ace05",
    "ace05_event": "ace-event",
    "mechanic_coarse": "None",
    "mechanic_granular": "covid-event",
}

origin_folder = "/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/ORKG_parsers/dygiepp/texts"

# Iterate over the dataset_codes and run the command for each one
for prediction_filename, dataset_code in dataset_codes.items():
    print(
        f"Processing {prediction_filename} with dataset code {dataset_code}...")
    subprocess.run(["python3.8", "ORKG_parsers/dygiepp/scripts/new-dataset/format_new_dataset.py",
                   origin_folder, f"ORKG_parsers/dygiepp/prepared_datasets/correct_format_{prediction_filename}", dataset_code, "--use-scispacy"])
