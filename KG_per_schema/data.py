'''Creates the concatenated set of sentences in data.txt. Subsequently uses this set to create the specific dataset for each particular DYGIE schema.'''

from datasets import load_dataset
import pickle
from utils import project_path
import re
# Helper functions
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# No need to move to utils since they are only used for initial sentence data loading


def is_research_sent(item, mode='OR'):
    '''Whether the sentence is a direction or a challenge. If mode is AND it returns only sentences that are both challenges and directions'''
    if mode == 'OR':
        return item['label'][0] > 0 or item['label'][1] > 0
    else:
        return item['label'][0] > 0 and item['label'][1] > 0


def pickle_group_info(item):
    '''Saves the context sentences to a pickled dict so that later can be retrieved what sentences belong to what context. This is needed since the dygie pipeline can only handle sentences and not texts (consisting of multiple sentences). With this group info the contexts can be recreated later from the sentences.'''
    group_info_fpath = project_path + \
        '/data/group_info/group_info.pkl'

    # Load the current dict
    with open(group_info_fpath, 'rb') as f:
        group_info = pickle.load(f)

    sents = [item['prev_sent'], item['text'], item['next_sent']]

    for sent in sents:
        group_info[clean_preprocess(sent)] = len(group_info)

    with open(group_info_fpath, 'wb') as f:
        pickle.dump(group_info, f)


def get_sents(item, context=True):
    '''Get the sentence + the sentence before and after it if context is true.'''
    if context:
        return [item['prev_sent'], item['text'], item['next_sent']]
    else:
        return [item['text']]


def clean_preprocess(sent):
    '''Making the sentence correct format in whitespace and newlines for dygie usage'''
    s = ' '.join(sent.replace('\n', ' ').replace(
        '\\', '').strip().split()) + ' '
    return re.sub(r'[^A-Za-z0-9 ,.-]+', '', s)


def opt_nl(sent, sentences):
    '''Always adds a newline except for the last sentence'''
    return '\n' if sent != sentences[-1] else ''


def clear_group_info(group_info_fpath):
    '''Empties the group info file when using context.'''
    with open(group_info_fpath, 'wb') as f:
        pickle.dump({}, f)

# Main code
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


def create_sentence_datasets():
    group_info_fpath = project_path + '/KG_per_schema/data/group_info/group_info.pkl'

    # Load dataset from Huggingface
    data = load_dataset(
        "DanL/scientific-challenges-and-directions-dataset", split="all")

    # Save 4 datasets (OR, AND challenges) x (with, without context)
    for mode in ['OR', 'AND']:
        for context in [True, False]:
            # If using the context (sentences around), the info about grouping needs to be noted
            if context:
                # Clean current group info
                clear_group_info(group_info_fpath)

                # Create group info
                [pickle_group_info(item) for item in data]

            # Get the relevant items + context
            sentences = [get_sents(item, context)
                         for item in data if is_research_sent(item, mode)]

            # Flatten the lists of context sents
            sentences = [s for l in sentences for s in l]

            # Clean sentences
            sentences = [clean_preprocess(sent) for sent in sentences]

            # Clear malformed sentences (some context sents are only a reference number, etc.)
            sentences = [sent for sent in sentences if len(
                sent.strip().split()) > 4]

            # Save
            postfix = ('context_' if context else '') + mode
            sents_target_fpath = f'KG_per_schema/data/sents/sents_{postfix}.txt'

            # Write sents with newlines to file (but not the last one)
            with open(sents_target_fpath, 'w') as f:
                [f.write(sent) for sent in sentences]


if __name__ == '__main__':
    create_sentence_datasets()
