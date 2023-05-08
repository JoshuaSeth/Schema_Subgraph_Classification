'''Builds the pre-parsed Dsouza NER from: https://github.com/jd-coderepos/cl-titles-parser.
This pre-parsed data is used in the streamlit app.'''

from ORKG_parsers.cl_titles_parser.thesis_interface import parse_cl
from datasets import load_dataset
import pickle

dataset = load_dataset(
    "DanL/scientific-challenges-and-directions-dataset", split="dev")

sentences = []
for item in dataset:
    if item['label'][0] > 0 and item['label'][1] > 0:
        sentences.append(item['text'])


ent_sents, rel_sents = parse_cl(sentences)

with open('dsouza_ents', 'wb') as f:
    pickle.dump(ent_sents, f)
