'''Creates the concatenated set of sentences in data.txt. Subsequently uses this set to create the specific dataset for each particular DYGIE schema.'''

from datasets import load_dataset

# Helper functions
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


def is_research_sent(item, mode='OR'):
    '''Whether the sentence is a direction or a challenge. If mode is AND it returns only sentences that are both challenges and directions'''
    if mode == 'OR':
        return item['label'][0] > 0 or item['label'][1] > 0
    else:
        return item['label'][0] > 0 and item['label'][1] > 0


def get_sent(item, context=True):
    '''Get the sentence + the sentence before and after it if context is true.'''
    if context:
        return item['prev_sent'] + ' ' + item['text'] + ' ' + item['next_sent']
    else:
        return item['text']


def clean_preprocess(sent):
    '''Making the sentence correct format in whitespace and newlines for dygie usage'''
    return " ".join(sent.replace('.', '. ').replace(
        '\n', ' ').replace('\\', '').strip().split()) + '\n'


# Main code
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# Load dataset from Huggingface
data = load_dataset(
    "DanL/scientific-challenges-and-directions-dataset", split="all")

# Get the relevant items + context
sentences = [get_sent(item) for item in data if is_research_sent(item)]

# Clean sentences
sentences = [clean_preprocess(sent) for sent in sentences]

# Save
with open('research_sents.txt', 'w') as f:
    [f.write(sent) for sent in sentences]
