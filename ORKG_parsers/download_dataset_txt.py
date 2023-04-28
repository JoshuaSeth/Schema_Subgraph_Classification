from datasets import load_dataset

dataset = load_dataset("DanL/scientific-challenges-and-directions-dataset", split="dev")

sentences = []
for item in dataset:
    if item['label'][0] > 0 and item['label'][1] > 0:
        sentences.append(item['text'])

with open('tom_hope_future_research.txt', 'w') as f:
    f.writelines(sentences)