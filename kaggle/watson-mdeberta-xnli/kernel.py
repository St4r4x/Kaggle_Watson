"""Kaggle kernel: fine-tune MoritzLaurer/mDeBERTa-v3-base-mnli-xnli on the competition data.
Standalone script — see kaggle/watson-kfold-xnli/kernel.py for why this vendors its own
NLIDataset/compute_metrics instead of importing this repo's src/ package.

mDeBERTa's native id2label (0=entailment,1=neutral,2=contradiction) already matches
Kaggle's own label order (verified via the Hub config.json) — no label remap needed here,
unlike the xlm-roberta-large-xnli kernel.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

COMPETITION_SLUG = "contradictory-my-dear-watson"
DATA_DIR = f"/kaggle/input/competitions/{COMPETITION_SLUG}"
OUTPUT_DIR = "/kaggle/working/outputs"

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1.409e-5
WARMUP_RATIO = 0.115
WEIGHT_DECAY = 0.087
GRAD_ACCUM_STEPS = 2
SEED = 42
VAL_SPLIT = 0.15


class NLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["premise"],
            row["hypothesis"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = int(row["label"])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7, (
        "GPU T4x2 required — select it in the Kaggle notebook's Session options "
        "before running (P100, sm_60, is not supported by the current torch wheel)"
    )

    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    print(f"Training samples: {len(df)}")

    df_train, df_val = train_test_split(
        df, test_size=VAL_SPLIT, stratify=df["label"], random_state=SEED
    )
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label2id = {v: k for k, v in ID2LABEL.items()}

    def model_init(trial=None):
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=label2id,
            torch_dtype=torch.float32,
        )

    train_dataset = NLIDataset(df_train, tokenizer, MAX_LENGTH)
    val_dataset = NLIDataset(df_val, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    metrics = trainer.evaluate()
    print(f"\nFinal val metrics: {metrics}")


if __name__ == "__main__":
    main()
