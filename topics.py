
import json
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import expit

MODEL = f"cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


model = AutoModelForSequenceClassification.from_pretrained(MODEL)
class_mapping = model.config.id2label

all_topics = []


def load_json(path):
    with open(path) as f:
        data = [json.loads(line) for line in tqdm(f.readlines())]
    return data


debaters = load_json("debaters/debaters.jsonl")

all_scores = []

for i, d in tqdm(enumerate(debaters)):
    print("\nRunning debater ", i, "comments")
    for c in tqdm(d["comments"]):
        text = c["text"]
        tokens = tokenizer(text, return_tensors='pt')
        output = model(**tokens)
        scores = output[0][0].detach().numpy()
        all_scores.append(scores)

all_scores = np.array(all_scores)
print(all_scores)
np.save('all_scores.npy', all_scores)
