'''Interface for the KG_per_schema module.'''
import streamlit as st
from streamlit_ui.sents_ui import viz_sents_ui
from streamlit_ui.graph_ui import viz_graph_ui
from streamlit_ui.graph_stats_ui import viz_graph_stats_ui
from streamlit_ui.encyclo_ui import viz_encyclo_ui

# Move to utils
def map_schema_names(selected_schemas):
    renamed_schemas = []
    for schema in selected_schemas:
        if 'spacy' in schema:
            renamed_schemas.append(schema.replace('_', '').replace(' ', '-'))
        else: renamed_schemas.append(schema)
    return renamed_schemas



# Variables
schemas = ['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event', 'spacy en_core_web_sm', 'spacy en_core_web_lg', 'spacy en_core_sci_scibert']
modes = ['AND', 'OR']
visualizations = ['sentences', 'graph', 'graph stats', 'encyclopedic explorer']

# Default interface options
st.header('KG Schema Explorer')

st.markdown('NOTE: To get the best performance when exporing schemas use the "AND" mode and disable using the context and only select a single schema. When you want the most complete interconnecte graph use the "OR" mode and enable using the context. You can combine multiple schemas into a single graph. Results are cached so the second time loading the results for a particular combination of parameters should be magnitudes faster.')
st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')

selected_schemas = st.multiselect('Schema', schemas)

# Rename the spacy schemas
renamed_schemas = map_schema_names(selected_schemas)

mode = st.selectbox('Mode', modes)
use_context = st.checkbox('Use context', value=True)
sent_tab, graph_tab, graph_stats_tab, ecyclo_tab = st.tabs(visualizations)

# Data flow state
if not 'current_ent' in st.session_state:
    st.session_state['current_ent'] = None
if not 'current_rel' in st.session_state:
    st.session_state['current_rel'] = None


def set_cur(ent=None, rel=None):
    '''Sets the current entity and relation to the given ones. If not given the entity or relation will be set to none.'''
    st.session_state['current_ent'] = ent
    st.session_state['current_rel'] = rel
    if ent == None:
        st.session_state.search_1 = 'relations'
    if rel == None:
        st.session_state.search_1 = 'entities'


if selected_schemas != None and renamed_schemas != [] and mode != None:
    # Visualize the sentences and the tagged entities and relations
    with sent_tab:
        viz_sents_ui(renamed_schemas, mode, use_context)

    # Visualize the full graph as an interactive graph
    with graph_tab:
        viz_graph_ui(renamed_schemas, mode, use_context)

    # Visualize the graph statistics for each schema
    with graph_stats_tab:
        viz_graph_stats_ui(schemas, mode, use_context)

    # Explore the graoh in encyclopedic fashion
    with ecyclo_tab:
        viz_encyclo_ui(renamed_schemas, mode, use_context, set_cur)
