import pickle
from annotated_text import annotated_text
import streamlit as st


def merge_words_and_entities(words: list, entities: list, sentence_start_idx: int) -> list:
    '''Function that takes turns a list of words and entities into a list of tuples or string where a string is simply a word and a tuple is a tagged span.
    The tagged span is a tuple of the span text and the label of the entity.
    words: list of words
    entities: list of entities
    sentence_start_idx: the index of the first word in the sentence in the entire text'''
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


def visualize_dygie_parser(data):
    '''Visualises the dygie parswer for the sentences with the chosen schema. According to: https://github.com/dwadden/dygiepp
    '''

    for idx, s in enumerate(data['sentences']):
        st.markdown('\n')
        st.subheader(str(idx))
        sent_start_idx = sum([len(sent)
                              for sent in data['sentences'][:idx]])
        words = [{'text': word, 'tag': ''} for word in s]

        if 'predicted_ner' in data:
            annotated_text(merge_words_and_entities(
                s, data['predicted_ner'][idx], sent_start_idx))

        if 'predicted_relations' in data:
            arcs = []
            for rel in data['predicted_relations'][idx]:
                sub = s[rel[0]-sent_start_idx:rel[1]-sent_start_idx+1]
                trigger = rel[4]
                obj = s[rel[2]-sent_start_idx:rel[3]-sent_start_idx+1]

                sub = ' '.join(sub)
                # ' '.join(arg1[0]) if len(arg1) > 0 else ''
                obj = ' '.join(obj)
                st.text(sub + ' '+trigger+' ' + obj)


def visualize_mechanic_granular_parser(data):
    '''Visualizes the data for the mechanic granular parser in steamlit sentences. Since the data is not in the same format as the other parsers, this function is a bit more complex than the default dygie parsers. See: visualize_dygie_parser'''
    st.markdown('Short explanation: The granular model of MECHANIC takes a word from the sentence to be the relation and sometimes knows which entities (the arg0 and arg1) this relation is between. Sometimes one or both of these entities is missing. Sometimes multiple arg0 or arg1 denote a coreference resolution between these words. Visualizes the data for the mechanic granular parser in steamlit sentences. Since the data is not in the same format as the other parsers, this function is a bit more complex than the default dygie parsers.')
    st.markdown(
        'For example for the first sentence (Molecular Tests, Detect, BVDV Isolates) and the second sentence has (Vaccines, Control, ___), where vaccines are coreferenced to be the same as Virological tests.')
    st.divider()
    for idx, s in enumerate(data['sentences']):
        st.markdown('\n')
        st.subheader(str(idx))
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
            print(temp_rel)

        if len(entities) > 0:
            entities = [item for sublist in entities for item in sublist]

            print('ents', entities)
            merged = merge_words_and_entities(
                s, entities, sent_start_idx)
            annotated_text(merged)
        else:
            st.text(' '.join(s))

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
                    arg0.append(
                        s[part[0] - sent_start_idx: part[1] - sent_start_idx+1])
                    fullarg0.append(part)
                elif part[2] == 'ARG1':
                    arg1.append(
                        s[part[0] - sent_start_idx: part[1] - sent_start_idx+1])
                    fullarg1.append(part)

            sub = ' '.join(arg0[0]) if len(arg0) > 0 else ' ? '
            obj = ' '.join(arg1[0]) if len(arg1) > 0 else ' ? '
            st.text(sub + ' - '+trigger+' - ' + obj)


def visualise_dsouza_parser():
    '''Creates a streamlit visualisation of: https://github.com/jd-coderepos/cl-titles-parser'''
    with open('ORKG_parsers/dsouza_ents', 'rb') as f:
        ent_sents = pickle.load(f)

    for idx, ent_sent in enumerate(ent_sents):
        st.markdown('\n')
        st.subheader(str(idx))
        annotated_text(ent_sent)
