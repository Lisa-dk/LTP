import json
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

def load_json(path):
    with open(path) as f:
        data = [json.loads(line) for line in tqdm(f.readlines())]
    return data

debaters = load_json("debaters/debaters.jsonl")

model_name = "nikolai40/human-values-roberta-aug"
tok_name = "roberta-base"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20)
tokenizer = AutoTokenizer.from_pretrained(tok_name)

def encode(str_input):
    return tokenizer(str_input, truncation=True)

results = []

for i, d in tqdm(enumerate(debaters)):
    print("\nRunning debater ", i, "comments")
    for c in tqdm(d["comments"]):
        text = c["text"]
        tokens = tokenizer(text, return_tensors='pt')

        # truncate
        if tokens["attention_mask"][0].shape[0] > 512:
            print("truncating tokens")
            tokens["input_ids"] = tokens["input_ids"][:,:512]
            tokens["attention_mask"] = tokens["attention_mask"][:,:512]

        output = model(**tokens)
        scores = output[0][0].detach().numpy()
        scores_softmax = np.exp(scores) / sum(np.exp(scores))
        results.append(scores_softmax)

results = np.array(results)
np.save("all_values.npy", results)