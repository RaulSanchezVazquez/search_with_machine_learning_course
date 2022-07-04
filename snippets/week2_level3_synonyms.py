"""
Then generate synonyms for these 1000 words. That’s easiest to do in python. Write code that:

Loads the fastText model you created in the previous step (and probably stored in workspace/datasets/fasttext/title_model.bin).

Iterates through each line of /workspace/datasets/fasttext/top_words.txt (or wherever you stored the top 1,000 title words).

Uses the model’s get_nearest_neighbors method to obtain each word’s nearest neighbors. Those are returned as an array of (similarity, word) pairs.

Outputs, for each word, a comma-separated line that starts with the word and is followed by the neighbors whose similarity exceeds a threshold. Try setting the threshold to be 0.75 or 0.8.
"""
import fasttext

MODEL_TH = 0.75
MODEL_PATH = '/workspace/datasets/fasttext/title_model.bin'
DATA_PATH = '/workspace/datasets/fasttext/top_words.txt'
OUTPUT_PATH = '/workspace/datasets/fasttext/synonyms.csv'

model = fasttext.load_model(MODEL_PATH)

with open(DATA_PATH, 'r') as f:
    data = f.read().split('\n')

data_synonyms = []
for word in data:
    synonyms = [
        synonym
        for (score, synonym) in model.get_nearest_neighbors(word)
        if score > MODEL_TH]

    data_synonyms.append(','.join([word] + synonyms))

with open(OUTPUT_PATH, 'w') as f:
    f.write('\n'.join(data_synonyms))
