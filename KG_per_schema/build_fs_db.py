'''Build a firebase firestore database from the results'''

from results_loader import build_encyclo_data
import firebase_admin
from firebase_admin import credentials, firestore, db
from utils import map_schema_names
from tqdm import tqdm
import time

# Init Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(
        "KG_per_schema/config/future-research-4fe7a-firebase-adminsdk-ahmte-95398f48a3.json")
    firebase_admin.initialize_app(cred)
fs = firestore.client()

# For all configurations
for schema in tqdm(map_schema_names(['scierc', 'spacy en_core_web_sm', 'spacy en_core_web_lg', 'spacy en_core_sci_scibert',
                                     'None', 'genia', 'covid-event', 'ace05', 'ace-event'])):
    for mode in ['OR', 'AND']:
        for use_context in [True, False]:
            config_name = f'{schema}_{mode}_{use_context}'

            doc_ref = fs.collection('schema_data').document(config_name)
            doc = doc_ref.get()
            if not doc.exists:
                print(f'Building {schema} {mode} {use_context}')

                ents, rels, sents_for_ents, rels_dict = build_encyclo_data(
                    [schema], mode, use_context)

                # First create the schema data doc to show entity list
                data = {'ents': []}
                for ent, val in ents.items():
                    data['ents'].append({'ref': ent, 'num rels': len(val)})

                try:
                    fs.collection('schema_data').document(
                        config_name).set(data)
                except:
                    time.sleep(20)
                    pass

                # Then for each entity a doc with the relations and sentences
                for ent, val in tqdm(ents.items(), leave=False):
                    data = {'rels': {}, 'sents': {}}
                    for idx, rel in enumerate(val):
                        data['rels'][str(idx)] = list(rel)

                    for idx, ent_sent in enumerate(sents_for_ents[ent]):
                        new_sent = []
                        for token in ent_sent:
                            if isinstance(token, tuple) or isinstance(token, list):
                                new_sent.append(
                                    {'word': token[0], 'tag': token[1]})
                            else:
                                new_sent.append({'word': token})
                        data['sents'][str(idx)] = new_sent

                    try:
                        fs.collection(config_name).document(ent).set(data)
                    except:
                        time.sleep(20)
                        pass
