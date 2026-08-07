"""
Training script for NLI classification (English-first).

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import glob
import os
import shutil
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
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


def hpo_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 3e-5, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
    }


class RDropTrainer(Trainer):
    """Two forward passes per batch (different dropout masks) + symmetric KL consistency loss.
    https://arxiv.org/abs/2106.14448
    """

    def __init__(self, *args, rdrop_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs1 = model(**inputs)
        outputs2 = model(**inputs)
        ce = 0.5 * (
            F.cross_entropy(outputs1.logits, labels)
            + F.cross_entropy(outputs2.logits, labels)
        )
        log_p1 = F.log_softmax(outputs1.logits, dim=-1)
        log_p2 = F.log_softmax(outputs2.logits, dim=-1)
        kl = 0.5 * (
            F.kl_div(log_p1, log_p2, log_target=True, reduction="batchmean")
            + F.kl_div(log_p2, log_p1, log_target=True, reduction="batchmean")
        )
        loss = ce + self.rdrop_alpha * kl
        return (loss, outputs1) if return_outputs else loss


def build_training_args(config: dict, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
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
        save_total_limit=1,
        save_only_model=True,
    )


def run_training(
    config: dict,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    model_init: Callable[..., PreTrainedModel],
    rdrop: bool,
    output_dir: str,
) -> dict:
    train_dataset = NLIDataset(df_train, tokenizer, config["max_length"])
    val_dataset = NLIDataset(df_val, tokenizer, config["max_length"])

    trainer_cls = RDropTrainer if rdrop else Trainer
    trainer_kwargs = {"rdrop_alpha": config.get("rdrop_alpha", 1.0)} if rdrop else {}
    trainer = trainer_cls(
        model=None,
        model_init=model_init,
        args=build_training_args(config, output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        **trainer_kwargs,
    )
    trainer.train()

    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"\nBest model saved to: {best_model_dir}")

    # best_model/ above already has everything downstream steps need — drop the raw
    # per-epoch checkpoint dirs so a --kfold run doesn't accumulate disk across folds.
    for checkpoint_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(checkpoint_dir)

    metrics = trainer.evaluate()
    print(f"\nFinal val metrics ({output_dir}): {metrics}")
    return metrics


def main(
    config: dict,
    hpo: bool = False,
    n_trials: int = 10,
    rdrop: bool = False,
    kfold: int | None = None,
) -> None:
    os.makedirs(config["output_dir"], exist_ok=True)

    df = load_train_data(
        config["data_dir"], config["language_filter"], config.get("label_map")
    )
    print(f"Training samples after filter: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model_init = build_model_init(config)

    if hpo:
        df_train, df_val = train_test_split(
            df,
            test_size=config["val_split"],
            stratify=df["label"],
            random_state=config["seed"],
        )
        train_dataset = NLIDataset(df_train, tokenizer, config["max_length"])
        val_dataset = NLIDataset(df_val, tokenizer, config["max_length"])
        trainer = Trainer(
            model=None,
            model_init=model_init,
            args=build_training_args(config, config["output_dir"]),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        best = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hpo_space,
            n_trials=n_trials,
            compute_objective=lambda metrics: metrics["eval_accuracy"],
        )
        print(f"\nBest trial: {best}")
        return

    if kfold:
        folds = make_stratified_folds(df, kfold, config["seed"])
        fold_accuracies = []
        for i, (train_idx, val_idx) in enumerate(folds):
            print(f"\n=== Fold {i + 1}/{kfold} ===")
            df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
            output_dir = os.path.join(config["output_dir"], f"fold_{i}")
            os.makedirs(output_dir, exist_ok=True)
            metrics = run_training(
                config, df_train, df_val, tokenizer, model_init, rdrop, output_dir
            )
            fold_accuracies.append(metrics["eval_accuracy"])
        avg_acc = sum(fold_accuracies) / len(fold_accuracies)
        print(f"\nAverage accuracy across {kfold} folds: {avg_acc:.4f}")
        return

    df_train, df_val = train_test_split(
        df,
        test_size=config["val_split"],
        stratify=df["label"],
        random_state=config["seed"],
    )
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")
    run_training(config, df_train, df_val, tokenizer, model_init, rdrop, config["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna hyperparameter search instead of a normal training run")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--rdrop", action="store_true", help="Use R-Drop regularization (two forward passes + KL consistency loss)")
    parser.add_argument("--kfold", type=int, default=None, help="Train N stratified folds instead of a single train/val split; saves each to outputs/fold_{i}/best_model")
    args = parser.parse_args()
    main(
        load_config(args.config),
        hpo=args.hpo,
        n_trials=args.n_trials,
        rdrop=args.rdrop,
        kfold=args.kfold,
    )
