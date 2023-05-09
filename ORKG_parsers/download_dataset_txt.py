from datasets import load_dataset

dataset = load_dataset(
    "DanL/scientific-challenges-and-directions-dataset", split="dev")

sentences = []
for item in dataset:
    print('\n\n', item)
    if item['label'][0] > 0 or item['label'][1] > 0:
        sentences.append(item['text'])

with open('hope.txt', 'w') as f:
    for sent in sentences:
        f.write(sent.replace('.', '. ').replace('\n', ' ') + '\n')
