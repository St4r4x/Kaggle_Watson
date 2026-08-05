"""
Training script for NLI classification (English-first).

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from dataset import NLIDataset, load_train_data, ID2LABEL


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def build_model_init(config: dict):
    id2label = config.get("checkpoint_id2label", ID2LABEL)
    label2id = {v: k for k, v in id2label.items()}

    def model_init(trial=None):
        return AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
            torch_dtype=torch.float32,  # from_pretrained silently inherits fp16 from the hub checkpoint's dtype metadata otherwise
        )

    return model_init


def hpo_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 3e-5, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
    }


def main(config: dict, hpo: bool = False, n_trials: int = 10):
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load data
    df = load_train_data(
        config["data_dir"], config["language_filter"], config.get("label_map")
    )
    print(f"Training samples after filter: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    df_train, df_val = train_test_split(
        df,
        test_size=config["val_split"],
        stratify=df["label"],
        random_state=config["seed"],
    )
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    train_dataset = NLIDataset(df_train, tokenizer, config["max_length"])
    val_dataset = NLIDataset(df_val, tokenizer, config["max_length"])

    model_init = build_model_init(config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        bf16=config["bf16"] and torch.cuda.is_available(),
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=True,
        logging_steps=50,
        report_to="mlflow",
        seed=config["seed"],
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

    if hpo:
        best = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hpo_space,
            n_trials=n_trials,
            compute_objective=lambda metrics: metrics["eval_accuracy"],
        )
        print(f"\nBest trial: {best}")
        return

    trainer.train()

    # Save final model and tokenizer
    best_model_dir = os.path.join(config["output_dir"], "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"\nBest model saved to: {best_model_dir}")

    # Final eval report
    metrics = trainer.evaluate()
    print(f"\nFinal val metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna hyperparameter search instead of a normal training run")
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()
    main(load_config(args.config), hpo=args.hpo, n_trials=args.n_trials)
