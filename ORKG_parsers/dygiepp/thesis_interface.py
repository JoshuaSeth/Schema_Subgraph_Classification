import spacy
from dygie.spacy_interface.spacy_interface import DygieppPipe
# import spacy_streamlit
import streamlit as st
# from spacy_streamlit import visualize_parser, visualize_ner
from spacy.tokens import Doc
import copy
import en_core_web_sm
from spacy import displacy
from datasets import load_dataset
from spacy_streamlit import visualize_ner
from annotated_text import annotated_text
import pickle

def parse_dygie(sentences, model_name='scierc'):
    
    sentences_entity_lists = []
    sentences_relation_arcs = []

    nlp = en_core_web_sm.load()

    component = DygieppPipe(nlp,pretrained_filepath=f"/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/ORKG_parsers/dygiepp/pretrained/{model_name}.tar.gz", dataset_name=model_name)
    nlp.add_pipe(component)

    # nlp.add_pipe(nlp.create_pipe('sentencizer'), before="dygiepp")

    for sentence in sentences:
        doc = nlp(sentence)

        words = [t.text for t in nlp(sentence)]
        spaces = [t.whitespace_ != "" for t in nlp(sentence)]
        custom_doc = Doc(nlp.vocab, words=words, spaces=spaces)

        # Copy sentence boundaries from the original doc to the custom_doc
        for sent in doc.sents:
            custom_doc[sent.start].is_sent_start = True

        # Define custom relations as (subject, object, relation_type) tuples
        custom_relations = list(doc._.rels)


        allrels= []
        # Set the head and dependency relation for each custom relation
        for sent in custom_relations:
            for rel in sent:
                subject, obj, rel_type = rel
                allrels.append(rel)


        words = [{"text": token.text, "tag": token.tag_} for token in custom_doc]
        arcs = []

        for sent in custom_relations:
            for rel in sent:
                subject, obj, rel_type = rel

                subject_idx = subject.start
                object_idx = obj.start

                if subject_idx < object_idx:
                    arcs.append({"start": subject_idx, "end": object_idx, "label": rel_type, "dir": "right"})
                else:
                    arcs.append({"start": object_idx, "end": subject_idx, "label": rel_type, "dir": "left"})

        word_entity_list = []
        sent_words = sentence.split()

        i = 0
        while i < len(sent_words):
            word = sent_words[i]
            found_entity = False
            for ent in doc.ents:
                span, label = ent.text, ent.label_
                if word in span:
                    word_entity_list.append((span, label))
                    # Increment the index by the number of words in the span
                    i += len(span.split())
                    found_entity = True
                    break

            if not found_entity:
                word_entity_list.append(word + ' ')
                i += 1


        sentences_entity_lists.append(word_entity_list)
        sentences_relation_arcs.append({"words": words, "arcs": arcs})
        
        with open ('/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/ORKG_parsers/dygie_data/' + model_name + '_rels', 'wb') as f:
            pickle.dump(sentences_relation_arcs, f)
        
        with open ('/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/ORKG_parsers/dygie_data/' + model_name +'_ents', 'wb') as f:
            pickle.dump(sentences_entity_lists, f)

        # svg = displacy.render({"words": words, "arcs": arcs}, style="dep", manual=True, options={'comact':True, "offset_x": 100, "distance": 100})

        # st.write(svg, unsafe_allow_html=True)


    return None,  {"words": words, "arcs": arcs}


dataset = load_dataset("DanL/scientific-challenges-and-directions-dataset", split="dev")
sentences = []
for item in dataset:
    if item['label'][0] > 0 and item['label'][1] > 0:
        sentences.append(item['text'])


for model_name in ['genia']:#, 'ace05-event','mechanic-granular', 'genia', 'mechanic-coarse', 'ace05-relation' , ]:
    parse_dygie(sentences, model_name=model_name)