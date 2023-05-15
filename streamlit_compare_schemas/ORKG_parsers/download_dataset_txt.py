from datasets import load_dataset

dataset = load_dataset(
    "DanL/scientific-challenges-and-directions-dataset", split="dev")

sentences = []
for item in dataset:
    print('\n\n', item)
    if item['label'][0] > 0 or item['label'][1] > 0:
        sentences.append(item['prev_sent'] + ' ' +
                         item['text'] + ' ' + item['next_sent'])

with open('hope.txt', 'w') as f:
    # We must replace the dots with comma's because they are turned into empty sentences making the models crash when using the context (with empty sentences)
    for sent in sentences:
        print(sent.replace('.', '. ').replace(
            '\n', ' ').replace('\\', '') + '\n')
        f.write(" ".join(sent.replace('.', '. ').replace(
            '\n', ' ').replace('\\', '').strip().split()) + '\n')
