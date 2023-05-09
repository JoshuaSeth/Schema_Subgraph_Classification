'''Testing hearst patternns library.  Can only be used with <=python3.10'''
from pyhearst import PyHearst
import nltk

nltk.download('averaged_perceptron_tagger')

ph = PyHearst()
text = 'works by such individuals as Marti A. Hearst, P. J. Proudhon, and Esther Duflo'
for pair in ph.extract_patterns(text):
    print(pair)
