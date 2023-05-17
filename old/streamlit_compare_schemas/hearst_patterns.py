'''Testing hearst patternns library.'''
from pyhearst import PyHearst
import nltk

nltk.download('averaged_perceptron_tagger')
ph = PyHearst()

with open('ORKG_parsers/dygiepp/texts/hope.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    print('\n', line)

    for pair in ph.extract_patterns(line):
        print(pair)
