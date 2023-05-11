import pandas as pd

df = pd.read_csv('research.csv')

df = df[(df['direction'] == 1) & (df['problem'] == 1)]

for line in df['text'].to_list():
    print(line)
    print('\n')
