import numpy as np
import pandas as pd
from datasets import Dataset
import json
import torch
import os

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer
)

df = pd.read_csv("sampled.csv")
shadow_dataset = Dataset.from_pandas(df)

tokenizer = DistilBertTokenizerFast.from_pretrained(r"C:\Users\jli93\Desktop\MIALM_project\victim_model_distilbert_agnews")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

N_SHADOW_MODELS = 5
split_dir = r"C:\Users\jli93\Desktop\MIALM_project\shadow_splits"
save_dir = r"C:\Users\jli93\Desktop\MIALM_project\shadow_models"

for i in range(N_SHADOW_MODELS):

    print(f"Training Shadow Model {i}")

    rng = np.random.default_rng(seed=42 + i)
    indices = np.arange(len(shadow_dataset))
    rng.shuffle(indices)

    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    #save split for later membership labeling
    split_path = f"{split_dir}/split_{i}.json"
    with open(split_path, "w") as f:
        json.dump({
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist()
        }, f)
    print(f"Saved split to {split_path}")

    #create split datasets
    train_dataset = shadow_dataset.select(train_indices)
    test_dataset  = shadow_dataset.select(test_indices)

    # Tokenize
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test  = test_dataset.map(tokenize, batched=True)

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test  = tokenized_test.rename_column("label", "labels")

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    #load a fresh DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4
    )

    #training Arguments
    training_args = TrainingArguments(
        output_dir=f"{save_dir}/shadow_model_{i}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    #train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    trainer.train()

    #save the trained shadow model
    model_save_path = f"{save_dir}/shadow_model_{i}/final"

    os.makedirs(model_save_path, exist_ok=True)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Shadow model {i} saved at: {model_save_path}")