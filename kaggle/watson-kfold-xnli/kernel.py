"""Kaggle kernel: 5-fold fine-tune of xlm-roberta-large-xnli on the competition data.
Standalone script — Kaggle kernels don't have this repo's src/ package layout, so the
minimal needed pieces of dataset.py/train.py are vendored here directly.
"""

import glob
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
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

MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
# joeddav checkpoint's native label order is 0=contradiction,1=neutral,2=entailment,
# the reverse of Kaggle's 0=entailment,1=neutral,2=contradiction.
CHECKPOINT_ID2LABEL = {0: "contradiction", 1: "neutral", 2: "entailment"}
LABEL_MAP = {0: 2, 1: 1, 2: 0}

MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1.409e-5
WARMUP_RATIO = 0.115
WEIGHT_DECAY = 0.087
GRAD_ACCUM_STEPS = 2
SEED = 42
N_FOLDS = 5


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


def make_stratified_folds(
    df: pd.DataFrame, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(df, df["label"]))
    all_val_idx = np.concatenate([val_idx for _, val_idx in folds])
    assert len(all_val_idx) == len(set(all_val_idx)) == len(df), (
        f"fold val indices must partition the full dataset exactly once: "
        f"got {len(all_val_idx)} indices, {len(set(all_val_idx))} unique, {len(df)} rows"
    )
    return folds


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7, (
        "GPU T4x2 required — select it in the Kaggle notebook's Session options "
        "before running (P100, sm_60, is not supported by the current torch wheel)"
    )

    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    df["label"] = df["label"].map(LABEL_MAP)
    print(f"Training samples: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label2id = {v: k for k, v in CHECKPOINT_ID2LABEL.items()}

    def model_init(trial=None):
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=CHECKPOINT_ID2LABEL,
            label2id=label2id,
            torch_dtype=torch.float32,
        )

    folds = make_stratified_folds(df, N_FOLDS, SEED)
    fold_accuracies = []
    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {i + 1}/{N_FOLDS} ===")
        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{i}")
        os.makedirs(fold_output_dir, exist_ok=True)

        train_dataset = NLIDataset(df_train, tokenizer, MAX_LENGTH)
        val_dataset = NLIDataset(df_val, tokenizer, MAX_LENGTH)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
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
            save_only_model=True,
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

        best_model_dir = os.path.join(fold_output_dir, "best_model")
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

        # Kaggle's disk is small relative to 5 folds x 5 epochs of xlm-roberta-large
        # checkpoints — the best_model/ copy above is all downstream steps need, so
        # drop the raw per-epoch checkpoint dirs now instead of letting them pile up
        # across folds.
        for checkpoint_dir in glob.glob(os.path.join(fold_output_dir, "checkpoint-*")):
            shutil.rmtree(checkpoint_dir)

        metrics = trainer.evaluate()
        print(f"Fold {i} val metrics: {metrics}")
        fold_accuracies.append(metrics["eval_accuracy"])

    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\nAverage accuracy across {N_FOLDS} folds: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
