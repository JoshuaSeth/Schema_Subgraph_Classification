'''Loads the results from the predictions repo. Gives back entities and relations. The dygie results might be in non-unfiform dictionaries. Which is the reason for this interface.'''


def get_entities(schema: str, mode: str = 'AND', context: bool = False):
    '''Returns a list of entities.
    mode: AND or OR'''
