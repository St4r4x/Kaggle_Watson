"""
Continue pretraining the competition's base checkpoint on the larger multilingual XNLI
corpus (facebook/xnli — ~392.7k pairs machine-translated into each of the competition's
15 languages) before the final fine-tune on train.csv's ~12k pairs.

Runs locally, not on Kaggle — needs the full XNLI download, too slow to refetch per
Kaggle kernel session.

Usage:
    python src/continued_pretrain.py --config configs/config.yaml \
        --output_dir outputs/xnli_pretrained --samples_per_language 20000
"""

import argparse
import os

from datasets import concatenate_datasets, load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from dataset import NLIDataset
from train import build_model_init, load_config

# Matches data/train.csv's lang_abv values exactly (verified) and facebook/xnli's
# per-language config names.
LANGUAGES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
XNLI_ROWS_PER_LANGUAGE = 392_702  # facebook/xnli "train" split size, same for every language config


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def load_xnli_sample(samples_per_language: int, seed: int):
    per_lang = [
        load_dataset("facebook/xnli", lang, split=f"train[:{samples_per_language}]")
        for lang in LANGUAGES
    ]
    combined = concatenate_datasets(per_lang).shuffle(seed=seed)
    print(
        f"Loaded {len(combined)} XNLI pairs "
        f"({samples_per_language}/language x {len(LANGUAGES)} languages, "
        f"out of {XNLI_ROWS_PER_LANGUAGE}/language available)"
    )
    return combined.to_pandas()


def main(config: dict, samples_per_language: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = load_xnli_sample(samples_per_language, config["seed"])
    # facebook/xnli's label order (0=entailment,1=neutral,2=contradiction) matches Kaggle's
    # own order. Remap to the checkpoint's native order the same way the competition
    # fine-tune does, so the classification head stays aligned throughout.
    label_map = config.get("label_map")
    if label_map:
        df["label"] = df["label"].map(label_map)

    n_val = min(5000, len(df) // 10)
    df_train, df_val = df[:-n_val], df[-n_val:]
    print(f"XNLI pretrain — train: {len(df_train)} | val: {len(df_val)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset = NLIDataset(df_train, tokenizer, config["max_length"])
    val_dataset = NLIDataset(df_val, tokenizer, config["max_length"])

    model_init = build_model_init(config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        bf16=config["bf16"] and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=100,
        report_to="mlflow",
        seed=config["seed"],
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nXNLI-continued-pretrained checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output_dir", default="outputs/xnli_pretrained")
    parser.add_argument("--samples_per_language", type=int, default=20000)
    args = parser.parse_args()
    main(load_config(args.config), args.samples_per_language, args.output_dir)
