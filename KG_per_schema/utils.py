'''Utility functions used by multiple scripts in folder'''

from os.path import dirname
import os

# Move to utils
def map_schema_names(selected_schemas):
    renamed_schemas = []
    for schema in selected_schemas:
        if 'spacy' in schema:
            renamed_schemas.append(schema.replace('_', '').replace(' ', '-'))
        else: renamed_schemas.append(schema)
    return renamed_schemas

def is_docker():
    path = '/proc/self/cgroup'
    FOUND_DOCKER = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)

    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any(
            'docker' in line for line in open(path)) or FOUND_DOCKER
    )


# Different for local running and runnning from docker
project_path: str = '/app' if is_docker() else dirname(dirname(__file__))


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
