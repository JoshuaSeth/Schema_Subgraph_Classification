'''Quick script to test whether a very simple rule based filtering of the DBPedia abstract leads to an alright alignment of text-triple pairs (or multi-text to multi-triple?)'''
import pandas as pd

def filter_abstract(abstract, w1, w2):
    new_abstract = ''
    for sentence in abstract.split('.'):
        if w1.replace('_', ' ') in sentence or w2.replace('_', ' ') in sentence:
            new_abstract += sentence + '. '
    return new_abstract

df = pd.read_csv('DBPedia_toy_data.csv')

df['object'] = df['object'].str.replace('http://dbpedia.org/resource/', '')
df['subject'] = df['subject'].str.replace('http://dbpedia.org/resource/', '')

# Use axis=1 to pass the entire row to the lambda function
df['abstract_sub'] = df.apply(lambda row: filter_abstract(row['abstract_sub'], row['subject'], row['object']), axis=1)
df['abstract_obj'] = df.apply(lambda row: filter_abstract(row['abstract_obj'], row['subject'], row['object']), axis=1)

print(df)

df.to_csv( 'filter_test.csv', index=False)