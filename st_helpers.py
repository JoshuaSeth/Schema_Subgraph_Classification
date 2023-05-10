import pickle
from annotated_text import annotated_text
import streamlit as st
from pyhearst import PyHearst
import nltk


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
        print(start, end, sentence_start_idx, len(merged))
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


class SchemaParser:
    '''Given a sentence and a schema, will tag entities and parse relations according to the schema.'''

    def __init__(self) -> None:
        nltk.download('averaged_perceptron_tagger')
        self.ph = PyHearst()

    def get_hearst_patterns(self, sentence: str) -> str:
        '''Given a sentence, will return the sentence with the Hearst patterns'''
        return self.ph.extract_patterns(' '.join(sentence))

    def parse_ents(self, sentence: str, schema: str, data, idx: int, sent_start_idx: int) -> list:
        '''sent_start_idx: index of the first word of the sentence in the text.'''
        def in_word_index(l, i):
            for idx, item in enumerate(l):
                i = i.split(' ')[0]
                if i in item:
                    return idx
            return None

        if schema == 'hearst':
            entities = self.get_hearst_patterns(sentence)
            if len(entities) > 0:
                print(entities, sentence)
            entities = [[in_word_index(sentence, ent[1]) + sent_start_idx, in_word_index(sentence,
                        ent[1])+len(ent[1].split())-1 + sent_start_idx, ent[0]] for ent in entities]
            return merge_words_and_entities(sentence, entities, sent_start_idx)
        if schema == 'granular':
            print(sentence)
            entities = process_granular_ents(
                data, idx, sentence, sent_start_idx)
            print('entities', entities)
            return merge_words_and_entities(
                sentence, entities, sent_start_idx)
        elif schema == "cl-titleparser":
            with open('ORKG_parsers/dsouza_ents', 'rb') as f:
                ent_sents = pickle.load(f)
            return ent_sents[idx]
        else:
            return merge_words_and_entities(
                sentence, data['predicted_ner'][idx], sent_start_idx)

    def parse_rels(self, schema,  data, sentence, idx, sent_start_idx):
        # Print the relation separately as (object, relation, subject)
        rels = []

        if schema == 'granular':
            for rel in data['predicted_events'][idx]:
                fullarg0 = []
                fullarg1 = []
                arg0 = []
                arg1 = []
                for part in rel:
                    if part[1] == 'TRIGGER':
                        trigger = sentence[part[0] - sent_start_idx]
                    elif part[2] == 'ARG0':
                        arg0.append(
                            sentence[part[0] - sent_start_idx: part[1] - sent_start_idx+1])
                        fullarg0.append(part)
                    elif part[2] == 'ARG1':
                        arg1.append(
                            sentence[part[0] - sent_start_idx: part[1] - sent_start_idx+1])
                        fullarg1.append(part)

                sub = ' '.join(arg0[0]) if len(arg0) > 0 else ' ? '
                obj = ' '.join(arg1[0]) if len(arg1) > 0 else ' ? '
                rels.append(sub + ' - '+trigger+' - ' + obj)
        elif schema == "cl-titleparser":
            pass
        elif schema == "hearst":
            pass
        else:
            if 'predicted_relations' in data:
                for rel in data['predicted_relations'][idx]:
                    sub = sentence[rel[0] -
                                   sent_start_idx:rel[1]-sent_start_idx+1]
                    trigger = rel[4]
                    obj = sentence[rel[2] -
                                   sent_start_idx:rel[3]-sent_start_idx+1]

                    sub = ' '.join(sub) if len(sub) > 0 else ' ? '
                    obj = ' '.join(obj) if len(obj) > 0 else ' ? '
                    rels.append(sub + ' - '+trigger+' - ' + obj)

        return rels


def visualize_dygie_parser(data):
    '''Visualises the dygie parswer for the sentences with the chosen schema. According to: https://github.com/dwadden/dygiepp
    '''

    for idx, s in enumerate(data['sentences']):
        st.markdown('\n')
        st.subheader(str(idx))
        sent_start_idx = sum([len(sent)
                              for sent in data['sentences'][:idx]])
        words = [{'text': word, 'tag': ''} for word in s]

        # Tag the relation arguments as entities (i.e. ("Covid-19 induced coughing:, "ARG0"))
        # And of course, tag the words with the found entities
        if 'predicted_ner' in data:
            annotated_text(merge_words_and_entities(
                s, data['predicted_ner'][idx], sent_start_idx))

        # Print the relation separately as (object, relation, subject)
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

        # Tag the relation arguments as entities (i.e. ("Covid-19 induced coughing:, "ARG0"))
        # Words have no entities here
        entities = process_granular_ents(data, idx, s, sent_start_idx)

        if len(entities) > 0:
            tagged_sentence = merge_words_and_entities(
                s, entities, sent_start_idx)
            annotated_text(tagged_sentence)
        else:
            st.text(' '.join(s))

        # Print the relation separately as (object, relation, subject)
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


def process_granular_ents(data, idx, s, sent_start_idx):
    '''Helper function for visualize_mechanic_granular_parser. Processes the events variable to a list of entities'''
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

    entities = [item for sublist in entities for item in sublist]
    return entities


def visualise_dsouza_parser():
    '''Creates a streamlit visualisation of: https://github.com/jd-coderepos/cl-titles-parser'''
    with open('ORKG_parsers/dsouza_ents', 'rb') as f:
        ent_sents = pickle.load(f)

    for idx, ent_sent in enumerate(ent_sents):
        st.markdown('\n')
        st.subheader(str(idx))
        annotated_text(ent_sent)
