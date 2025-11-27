import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = r"C:\Users\jli93\Desktop\MIALM_project\data\sampled.csv"
SHADOW_SPLIT_DIR = r"C:\Users\jli93\Desktop\MIALM_project\shadow_splits"
SHADOW_MODELS_DIR = r"C:\Users\jli93\Desktop\MIALM_project\shadow_models"

N_SHADOW_MODELS = 5

df = pd.read_csv(DATA_PATH)
dataset = Dataset.from_pandas(df)

texts = df["text"].tolist()

tokenizer = DistilBertTokenizerFast.from_pretrained(
    r"C:\Users\jli93\Desktop\MIALM_project\victim_model_distilbert_agnews"
)

#list of probability vectors
attack_features = []
#member = 1, non-member = 0
attack_labels = []

for i in range(N_SHADOW_MODELS):

    print(f"\nProcessing Shadow Model {i}")

    #load shadow model
    model_path = f"{SHADOW_MODELS_DIR}/shadow_model_{i}/final"

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    #load membership split info
    split_file = f"{SHADOW_SPLIT_DIR}/split_{i}.json"
    with open(split_file, "r") as f:
        split_info = json.load(f)

    train_ids = set(split_info["train_indices"])
    test_ids = set(split_info["test_indices"])
    all_ids = train_ids.union(test_ids)

    #process only the indices used by this shadow model
    for idx in tqdm(sorted(all_ids), desc=f"Shadow {i}"):

        text = texts[idx]

        #tokenize
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        encoding = {k: v.to(device) for k, v in encoding.items()}

        #forward pass, output signal used: softmax probabilities
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1).cpu().tolist()

        #determine membership label
        if idx in train_ids:
            label = 1
        else:
            label = 0

        attack_features.append(probs)
        attack_labels.append(label)

#save attack data
output_file = r"C:\Users\jli93\Desktop\MIALM_project\attack_data\attack_train_data.pkl"

with open(output_file, "wb") as f:
    pickle.dump({
        "X": attack_features,
        "y": attack_labels
    }, f)

print(f"\nSaved attack training data to: {output_file}")
print(f"Total samples: {len(attack_labels)}")
print("Member count:", sum(attack_labels))
print("Non-member count:", len(attack_labels) - sum(attack_labels))
