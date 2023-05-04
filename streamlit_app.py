
# import sys
# from pathlib import Path # if you haven't already done so
# file = Path(__file__).resolve()
# parent, root = file.parent, file.parents[1]
# sys.path.append(str(root))

# # Additionally remove the current file's directory from sys.path
# try:
#     sys.path.remove(str(parent))
# except ValueError: # Already removed
#     pass

from datasets import load_dataset
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

        start -= sentence_start_idx
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


dataset = load_dataset(
    "DanL/scientific-challenges-and-directions-dataset", split="dev")

sentences = []
for item in dataset:
    if item['label'][0] > 0 and item['label'][1] > 0:
        sentences.append(item['text'])


option = st.selectbox(
    'Joint NER and RE method',
    ("D'Souza's CL-TitleParser", 'Dygie SciErc', 'Dygie GENIA', 'Dygie Ace05_Rels', 'Dygie Ace05_Event', 'Dygie MECHANIC-coarse', 'Dygie MECHANIC-granular'))


st.header('NER & RE Parsing Results')
if option == "D'Souza's CL-TitleParser":
    with open('dsouza_ents', 'rb') as f:
        ent_sents = pickle.load(f)

    for ent_sent in ent_sents:
        annotated_text(ent_sent)

if "Dygie" in option:
    model = option.replace('MECHANIC-', '').replace('05',
                                                    '').split()[1].lower()
# The thesis interfaced can be run in the dygie repo but results are pickled for performance reasons
    with open(f'ORKG_parsers/dygiepp/predictions/{model}.jsonl', 'r') as f:
        data = f.read()

    data = eval(data)

    if not 'granular' in option.lower():
        for idx, s in enumerate(data['sentences']):
            sent_start_idx = sum([len(sent)
                                 for sent in data['sentences'][:idx]])
            words = [{'text': word, 'tag': ''} for word in s]

            if 'predicted_ner' in data:
                annotated_text(merge_words_and_entities(
                    s, data['predicted_ner'][idx], sent_start_idx))

            if 'predicted_relations' in data:
                arcs = []
                for rel in data['predicted_relations'][idx]:
                    print(s[rel[0]-sent_start_idx:rel[1]-sent_start_idx+1],
                          rel[4], s[rel[2]-sent_start_idx:rel[3]-sent_start_idx+1])
                    arcs.append({
                        "start": rel[0]-sent_start_idx,
                        "end": rel[2]-sent_start_idx,
                        "label": rel[4],
                        "dir": "right" if rel[0]-sent_start_idx < rel[2]-sent_start_idx else "left"
                    })
                try:
                    svg = displacy.render({'words': words, 'arcs': arcs}, style="dep", manual=True, options={
                        "offset_x": 100, "distance": 100})
                    st.write(svg, unsafe_allow_html=True)
                except Exception as e:
                    st.text(e)

    if 'granular' in option.lower():
        st.markdown('Short explanation: The granular model of MECHANIC takes a word from the sentence to be the relation and sometimes knows which entities (the arg0 and arg1) this relation is between. Sometimes one or both of these entities is missing. Sometimes multiple arg0 or arg1 denote a coreference resolution between these words.')
        st.markdown(
            'For example for the first sentence (Molecular Tests, Detect, BVDV Isolates) and the second sentence has (Vaccines, Control, ___), where vaccines are coreferenced to be the same as Virological tests.')
        for idx, s in enumerate(data['sentences']):
            sent_start_idx = sum([len(sent)
                                 for sent in data['sentences'][:idx]])
            words = [{'text': word, 'tag': ''} for word in s]

            print(s)

            # And the others as entities
            entities = []
            for rel in data['predicted_events'][idx]:
                temp_rel = []
                for part in rel:
                    if part[1] != 'TRIGGER':
                        temp_rel.append(part)
                    else:
                        temp_rel.append(
                            [part[0], part[0], s[part[0] - sent_start_idx]])
                entities.append(temp_rel)

            if len(entities) > 0:
                print('ents', entities[0])
                annotated_text(merge_words_and_entities(
                    s, entities[0], sent_start_idx))

            # If there is a trigger relation in the sentence
            # Mark the trigger as an entity

            arcs = []
            for rel in data['predicted_events'][idx]:
                print(rel)
                fullarg0 = []
                fullarg1 = []
                arg0 = []
                arg1 = []
                for part in rel:
                    if part[1] == 'TRIGGER':
                        trigger = s[part[0] - sent_start_idx]
                    elif part[2] == 'ARG0':
                        arg0.append(s[part[0] - sent_start_idx])
                        fullarg0.append(part)
                    elif part[2] == 'ARG1':
                        arg1.append(s[part[0] - sent_start_idx])
                        fullarg1.append(part)

                print(arg0, trigger, arg1)
                # Do rels with 2 parts as rels
                if len(arg0) > 0 and len(arg1) > 0:
                    arcs.append({
                        "start": fullarg0[0][0]-sent_start_idx,
                        "end": fullarg1[0][0]-sent_start_idx,
                        "label": trigger,
                        'dir': 'right' if fullarg0[0][0]-sent_start_idx < fullarg1[0][0]-sent_start_idx else 'left'
                    })

            svg = displacy.render({'words': words, 'arcs': arcs}, style="dep", manual=True, options={
                "offset_x": 100, "distance": 100})
            st.write(svg, unsafe_allow_html=True)
