
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from datasets import load_dataset
from ORKG_parsers.cl_titles_parser.thesis_interface import parse_cl
import pickle
from annotated_text import annotated_text
import streamlit as st
import subprocess
from spacy import displacy



def merge_words_and_entities(words, entities, sentence_start_idx):
    merged = [None] * len(words)
    for word_idx, word in enumerate(words):
        merged[word_idx] = word + ' '

    for entity in entities:
        start, end, label = entity[0], entity[1], entity[2]

        start-= sentence_start_idx
        if isinstance(end, str):
            label = end
            end = start+1
        end -= sentence_start_idx

        span_text = ' '.join(words[start:end + 1])

        for idx in range(start, end + 1):
            if idx == start:
                if isinstance(merged[idx], tuple):
                    merged[idx] = (span_text, label)
                else:
                    merged[idx] = (span_text, label)
            else:
                merged[idx] = None

    merged = [item for item in merged if item is not None]

    return merged



dataset = load_dataset("DanL/scientific-challenges-and-directions-dataset", split="dev")

sentences = []
for item in dataset:
    if item['label'][0] > 0 and item['label'][1] > 0:
        sentences.append(item['text'])


option = st.selectbox(
    'Joint NER and RE method',
    ( "D'Souza's CL-TitleParser", 'Dygie SciErc', 'Dygie GENIA', 'Dygie Ace05_Rels', 'Dygie Ace05_Event', 'Dygie MECHANIC-coarse', 'Dygie MECHANIC-granular'))


st.header('NER & RE Parsing Results')
if option == "D'Souza's CL-TitleParser":
    ent_sents, rel_sents= parse_cl(sentences)

    for ent_sent in ent_sents:  
        annotated_text(ent_sent)
            
if "Dygie" in option:
    model = option.replace('MECHANIC-', '').replace('05', '').split()[1].lower()
# The thesis interfaced can be run in the dygie repo but results are pickled for performance reasons
    with open(f'ORKG_parsers/dygiepp/predictions/{model}.jsonl', 'r') as f:
        data =f.read()

    data = eval(data)

    for idx, s in enumerate(data['sentences']):
        sent_start_idx = sum([len(sent) for sent in data['sentences'][:idx]])
        words = [{'text': word, 'tag': ''} for word in s]

        if 'predicted_ner' in data: 
            annotated_text(merge_words_and_entities(s, data['predicted_ner'][idx], sent_start_idx))

        if 'predicted_relations' in data:
            arcs = []
            for rel in data['predicted_relations'][idx]:
                print(s[rel[0]-sent_start_idx:rel[1]-sent_start_idx+1], rel[4], s[rel[2]-sent_start_idx:rel[3]-sent_start_idx+1])
                arcs.append({
                        "start": rel[0]-sent_start_idx,
                        "end": rel[2]-sent_start_idx,
                        "label": rel[4],
                        "dir": "right" if rel[0]-sent_start_idx < rel[2]-sent_start_idx else "left"
                    })
            svg = displacy.render({'words': words, 'arcs': arcs}, style="dep", manual=True, options={ "offset_x": 100, "distance": 100})
            st.write(svg, unsafe_allow_html=True)

