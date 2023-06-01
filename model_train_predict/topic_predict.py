
# Predict the topics of reddit comments using the cardiffnlp/tweet-topic-21-multi model
# See https://huggingface.co/cardiffnlp/tweet-topic-21-multi

import json
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np

MODEL = "cardiffnlp/tweet-topic-21-multi"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def load_json(path):
    with open(path) as f:
        data = [json.loads(line) for line in tqdm(f.readlines())]
    return data


debaters = load_json("debaters.jsonl")

all_scores = []

for i, d in tqdm(enumerate(debaters)):
    print("\nRunning debater ", i, "comments")
    for c in tqdm(d["comments"]):
        text = c["text"]
        tokens = tokenizer(text, return_tensors='pt')

        # truncate
        if tokens["attention_mask"][0].shape[0] > 512:
            print("truncating tokens")
            tokens["input_ids"] = tokens["input_ids"][:, :512]
            tokens["attention_mask"] = tokens["attention_mask"][:, :512]

        output = model(**tokens)
        scores = output[0][0].detach().numpy()
        all_scores.append(scores)

all_scores_np = np.array(all_scores)
np.save('all_scores.npy', all_scores_np)
