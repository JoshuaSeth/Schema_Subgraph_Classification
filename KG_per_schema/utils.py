'''Utility functions used by multiple scripts in folder'''

from os.path import dirname


project_path: str = dirname(dirname(__file__))


def get_model_fname(filename):
    '''Get the appropriate model name from the training file'''
    filename = filename.split('_')[0]
    if filename == "scierc":
        return "scierc.tar.gz"
    if filename == "genia":
        return "genia.tar.gz"
    if filename == "ace05":
        return "ace05-relation.tar.gz"
    if filename == "ace-event":
        return "ace05-event.tar.gz"
    if filename == "None":
        return "mechanic-coarse.tar.gz"
    if filename == "covid-event":
        return "mechanic-granular.tar.gz"
