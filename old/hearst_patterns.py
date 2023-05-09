'''Returns a list of hearst patterns found in a sentence.
Substantially modified from: https://github.com/kinivi/patent_ner_linking/blob/main/project.ipynb'''
import pandas as pd
from spacy.matcher import PhraseMatcher, Matcher
from spacy.util import filter_spans
from spacy import displacy
from spacy.tokens import DocBin
from spacy.tokens import Span
from collections import Counter
import spacy
import json
import sys


class Hearst_Patterns:
    """ Extracts hearst patterns from a corpus
    """

    def __init__(self, patterns_file="patterns.json", text_path="G06K.txt"):
        """ creates an instance of the class Hearst_Patterns

        Args:
            patterns_file (path, optional): the json file containing the patterns. Defaults to "patterns.json".
            model_path (path, optional): the folder containing the NER model to use. Defaults to "spacy/model-new".
            text_path (path, optional): the file containing the corpus analyse and extract patterns. Defaults to "G06K.txt".
        """

        # read the text file
        textfile = open(text_path).read().strip()
        self.sentences = textfile.split('\n')

        # load the models
        spacy.cli.download("en_core_web_lg")
        self.nlp = spacy.load("en_core_web_lg")
        self.en_nlp = spacy.load("en_core_web_lg")
        self.nlp.add_pipe("merge_entities")
        self.en_nlp.add_pipe('merge_noun_chunks')

        self.matcher = Matcher(self.nlp.vocab)
        self.patterns = self.load_patterns_from_json(patterns_file)
        for name, pattern in self.patterns:
            self.matcher.add(name, pattern)

        # this list is used in the method get_matches
        self.continue_words = [',', 'and', 'or', ';', 'also', 'as well']

    def load_patterns_from_json(self, patterns_file):
        """ read the json file and return the list of

        Args:
            patterns_file (path): the json file containing the patterns

        Returns:
            List: a list of the hearst patterns found in the json file  
        """
        f = open(patterns_file)
        data = json.load(f)
        patterns = []
        for name, pattern in data.items():
            patterns.append((name, pattern))

        return patterns

    def extract_patterns(self, size=10, save_folder=".", start=0):
        """ look for matches in a corpus (text file)

        Args:
            size (int, optional): the minimum number of matches to be found. Defaults to 10.
            save_folder (path, optional): the folder in which save the resulted csv file. Defaults to ".".
            start (int, optional): the first line in which we start to look for matches (useful to continue where you stopped). Defaults to 0.
        """
        extracted_patterns = []

        # chose a start
        line = start
        count = 0

        # for the output
        print(f'{count} pattern extracted...', end='\r')

        while count < size and line < len(self.sentences)-1:
            while True:  # we read line by line until finding a match, to keep track of the count
                try:  # it bugs very rarely, don't know why XD

                    # look for a match
                    patterns = self.get_matches(self.sentences[line])

                    if patterns:
                        extracted_patterns.append(patterns)
                        break
                    print(f'{count} patterns extracted...{line}', end='\r')

                except Exception as e:
                    print("An error has occurred", e)

                line += 1
            count = len(extracted_patterns)
            print(f'{count} patterns extracted...{line}', end='\r')

        print(f'({count}) patterns extracted from lines ({start}-{line}))')
        save_file = f"{save_folder}/hearst_patterns.{len(extracted_patterns)}.csv"
        print(f'Patterns saved to {save_file}')
        df = pd.DataFrame(extracted_patterns, columns=[
                          'word1', 'word2', 'relation', 'label', 'text'])
        df.to_csv(save_file)

    def get_matches(self, text):
        label = {
            'rhyper': -1,
            'hyper': 1,
        }
        # because patterns like < !(bla bla) X > don't work when X is in the beginning of the sentence
        doc = self.nlp('. '+text)

        matches = self.matcher(doc)
        relations = []
        print(matches)
        for match_id, start, end in matches:

            # get all entities indices in the doc
            ent_indices = [i for i in range(start, end) if doc[i].text in [
                ent.text for ent in doc[start:end].ents]]
            if not ent_indices:  # no entity found
                return []

            # extract X...Y from a match ..X...Y.., so now we know that the first and the last token are the entities
            span = doc[min(ent_indices):max(ent_indices)+1]

            # Get string representation
            match_info = self.nlp.vocab.strings[match_id]
            match_name = match_info.split('-')[0]   # hyper or rhyper
            match_type = match_info.split('-')[1]   # single or multi

            np_0 = span[0]  # left term
            np_1 = span[-1]  # right term (or first right term if multiple)

            # all the right terms (ex. for Y...X1, X2, ...Xn) X1...Xn are the right terms
            right_terms = [np_1.text]
            if match_type == "multi":  # look for other terms (X2,X3..etc)

                # we use the en_core_web_lg model to get the noun chunks
                doc_en = self.en_nlp(doc[end:].text)
                for d in doc_en:
                    # look for entities inside the noun chunk
                    matching_ents = [
                        ent.text for ent in doc.ents if ent.text in d.text]
                    if matching_ents:
                        right_terms.append(matching_ents[0])
                    elif d.text not in self.continue_words:  # stop when seeing a word that's not in the list
                        break

            for term in right_terms:
                relations.append(
                    (np_0.text, term, match_name, label[match_name], text))

        relations = set(relations)
        return list(relations)


hp = Hearst_Patterns(patterns_file="hearst_patterns.json",
                     text_path="ORKG_parsers/dygiepp/texts/hope.txt")
hp.extract_patterns(size=10, start=0, save_folder="extracted_patterns")

# df_results = pd.read_csv(
#     "extracted_patterns/hearst_patterns.155.csv", index_col=0)[:50]
# df_results[['Hypernym', 'Hyponym', 'Frequency']].head(10)
