# Kaggle Compute Offload + Ensembling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add K-fold training, a second base-model fine-tune, and probability-averaging ensembling — split across two Kaggle GPU kernels and the local desktop — to try to beat the current best result (92.24% val acc / 0.91241 public score).

**Architecture:** `src/train.py` gains a `--kfold N` flag (StratifiedKFold loop, reusing the existing `Trainer` setup). Two self-contained Kaggle kernel scripts (Kaggle kernels can't import this repo's `src/` layout, so they vendor the minimal needed code) train the K-fold XLM-R ensemble and a second `mDeBERTa-v3-base-mnli-xnli` model. A new `src/predict_ensemble.py` (and its Kaggle-kernel counterpart) loads every trained checkpoint, reads each one's own saved `id2label` to realign its output probabilities into a common label order, and averages them. A separate local-only `src/continued_pretrain.py` continues training the checkpoint on the full `facebook/xnli` corpus before the competition fine-tune, independent of the ensemble.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers/Datasets, scikit-learn, Kaggle CLI 2.2.4 (`kaggle kernels push`, `kaggle datasets create`).

## Global Constraints

- Kaggle username: `st4r4x` (from `kaggle config view`). Kernel/dataset ids use this owner.
- Competition slug: `contradictory-my-dear-watson`.
- No pytest in this repo (`requirements.txt` has no test framework) — verification steps below use plain `assert`-based smoke scripts run directly via `python`, per this project's existing convention (no test framework files, per [CLAUDE.md](../../../CLAUDE.md) and the ponytail convention already active in this session: one runnable check per non-trivial branch/loop, no test scaffolding).
- Kaggle's free-tier P100 GPU (compute capability sm_60) is not supported by the current PyTorch wheel — training kernels must use **GPU T4×2**, not P100 (same constraint already documented in [CLAUDE.md](../../../CLAUDE.md) for the existing inference kernel).
- `kaggle datasets create` defaults to `--dir-mode skip`, which silently drops subdirectories. Every dataset upload in this plan that includes nested folders (`fold_0/best_model/`, etc.) MUST pass `--dir-mode zip` or the upload will be empty of model weights.
- `joeddav/xlm-roberta-large-xnli`'s native label order is `0=contradiction,1=neutral,2=entailment` (reverse of Kaggle's `0=entailment,1=neutral,2=contradiction`). `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`'s native order is `0=entailment,1=neutral,2=contradiction` — **identical** to Kaggle's, verified via its `config.json` on the Hub. No `label_map`/`checkpoint_id2label` needed for the mDeBERTa config.
- Python style (per project rules): type hints on all signatures, f-strings, `pathlib.Path` over `os.path` where practical, stdlib → third-party → local import grouping (alphabetical within each group), no bare `except:`.

---

### Task 1: K-fold training support in `src/train.py`

**Files:**
- Modify: `src/train.py`

**Interfaces:**
- Produces: `make_stratified_folds(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]`, `build_training_args(config: dict, output_dir: str) -> TrainingArguments`, `run_training(config: dict, df_train: pd.DataFrame, df_val: pd.DataFrame, tokenizer, model_init, rdrop: bool, output_dir: str) -> dict` — all module-level functions in `src/train.py`, importable by `src/continued_pretrain.py` (Task 7) as `from train import build_model_init, load_config`.

- [ ] **Step 1: Write the failing check for fold partitioning**

Run this from the repo root (uses the venv's installed sklearn/numpy/pandas, no GPU needed):

```bash
python3 - <<'EOF'
import sys
sys.path.insert(0, "src")
import pandas as pd
from train import make_stratified_folds

df = pd.DataFrame({"label": [0, 1, 2] * 20})
folds = make_stratified_folds(df, n_splits=5, seed=42)
assert len(folds) == 5
print("OK")
EOF
```

Expected: `ImportError: cannot import name 'make_stratified_folds'` (function doesn't exist yet).

- [ ] **Step 2: Refactor `src/train.py`**

Replace the import line at the top (`src/train.py:12`):

```python
from sklearn.model_selection import train_test_split
```

with:

```python
from sklearn.model_selection import StratifiedKFold, train_test_split
```

Insert this new function right after `build_model_init` (after `src/train.py:55`, before `def hpo_space`):

```python
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
```

This needs `import pandas as pd` — add it to the top-level imports (`src/train.py:8-13` currently has `import numpy as np` but not pandas; add `import pandas as pd` alphabetically before it).

Now replace the entire body from `def main(...)` through the end of the file (`src/train.py:93-185`) with:

```python
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
    )


def run_training(
    config: dict,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer,
    model_init,
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
```

- [ ] **Step 3: Re-run the fold-partitioning check**

```bash
python3 - <<'EOF'
import sys
sys.path.insert(0, "src")
import pandas as pd
from train import make_stratified_folds

df = pd.DataFrame({"label": [0, 1, 2] * 20})
folds = make_stratified_folds(df, n_splits=5, seed=42)
assert len(folds) == 5
for train_idx, val_idx in folds:
    assert len(train_idx) + len(val_idx) == len(df)
print("OK")
EOF
```

Expected: `OK` printed, no assertion error.

- [ ] **Step 4: Confirm the module still imports and the CLI still parses**

```bash
python3 src/train.py --help
```

Expected: argparse help text listing `--config`, `--hpo`, `--n-trials`, `--rdrop`, `--kfold`, no traceback.

- [ ] **Step 5: Commit**

```bash
git add src/train.py
git commit -m "feat: add --kfold flag for stratified k-fold training"
```

---

### Task 2: `configs/config_mdeberta.yaml`

**Files:**
- Create: `configs/config_mdeberta.yaml`

**Interfaces:**
- Consumes: nothing (standalone config file)
- Produces: a config dict shape identical to `configs/config.yaml`'s, consumed by `src/train.py:load_config` and the Kaggle kernel in Task 5.

- [ ] **Step 1: Create the config**

```yaml
model_name: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli   # already fine-tuned on MNLI+XNLI; native id2label is 0=entailment,1=neutral,2=contradiction — identical to Kaggle's own order, verified via the Hub config.json, so no label_map/checkpoint_id2label remap is needed here (unlike config.yaml's xlm-roberta-large-xnli)

# Data
data_dir: data/
output_dir: outputs/mdeberta/
language_filter: all       # "english" or "all"
val_split: 0.15
seed: 42

# Tokenization
max_length: 256

# Training
batch_size: 8
num_epochs: 5
learning_rate: 1.409e-5   # reused from the Optuna sweep for xlm-roberta-large-xnli as a starting point; not separately tuned for mDeBERTa
warmup_ratio: 0.115
weight_decay: 0.087
gradient_accumulation_steps: 2
bf16: false

# Evaluation
metric_for_best_model: accuracy
load_best_model_at_end: true
eval_strategy: epoch
save_strategy: epoch
```

- [ ] **Step 2: Verify it parses and matches the expected shape**

```bash
python3 -c "
import yaml
c = yaml.safe_load(open('configs/config_mdeberta.yaml'))
assert c['model_name'] == 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
assert 'label_map' not in c
assert 'checkpoint_id2label' not in c
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add configs/config_mdeberta.yaml
git commit -m "feat: add mDeBERTa-v3-xnli training config"
```

---

### Task 3: `src/predict_ensemble.py` (local ensembling script)

**Files:**
- Create: `src/predict_ensemble.py`

**Interfaces:**
- Consumes: `NLIDataset`, `load_test_data`, `ID2LABEL` from `src/dataset.py` (existing, unchanged).
- Produces: `align_probs_to_canonical(probs: np.ndarray, id2label: dict[int, str]) -> np.ndarray` — pure function, also reimplemented (duplicated, since Kaggle kernels are standalone) in Task 6's kernel script with the same behavior.

This is the correctness-critical piece: `joeddav/xlm-roberta-large-xnli` checkpoints and the `mDeBERTa` checkpoint have **different native label orders**. Averaging their raw softmax outputs without realigning columns would silently corrupt every prediction. Each saved checkpoint's own `config.json` already records its true `id2label` (set at training time via `build_model_init`), so this reads that back rather than hardcoding which models need which remap.

- [ ] **Step 1: Write the failing check for label alignment**

```bash
python3 - <<'EOF'
import sys
sys.path.insert(0, "src")
import numpy as np
from predict_ensemble import align_probs_to_canonical

# xlm-roberta-large-xnli's native order: 0=contradiction,1=neutral,2=entailment
reversed_id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
probs = np.array([[0.1, 0.2, 0.7]])  # contradiction=0.1, neutral=0.2, entailment=0.7
aligned = align_probs_to_canonical(probs, reversed_id2label)
expected = np.array([[0.7, 0.2, 0.1]])  # entailment, neutral, contradiction
assert np.allclose(aligned, expected), f"{aligned} != {expected}"

# mDeBERTa's native order already matches Kaggle's: identity, no change
identity_id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
aligned2 = align_probs_to_canonical(probs, identity_id2label)
assert np.allclose(aligned2, probs), f"{aligned2} != {probs}"

print("OK")
EOF
```

Expected: `ModuleNotFoundError: No module named 'predict_ensemble'` (file doesn't exist yet).

- [ ] **Step 2: Write `src/predict_ensemble.py`**

```python
"""
Generate submission.csv by averaging softmax probabilities across multiple trained
model checkpoints (e.g. K-fold checkpoints + a second base model).

Usage:
    python src/predict_ensemble.py \
        --model_dirs outputs/fold_0/best_model outputs/fold_1/best_model outputs/mdeberta/best_model \
        --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from dataset import ID2LABEL, NLIDataset, load_test_data

CANONICAL_LABELS = [ID2LABEL[i] for i in range(3)]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def align_probs_to_canonical(probs: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    """Reorder a model's [N, 3] softmax output so column i is always CANONICAL_LABELS[i],
    regardless of the checkpoint's own native label ordering."""
    label_to_idx = {label: idx for idx, label in id2label.items()}
    missing = set(CANONICAL_LABELS) - set(label_to_idx)
    assert not missing, f"checkpoint id2label is missing labels: {missing} (got {id2label})"
    permutation = [label_to_idx[label] for label in CANONICAL_LABELS]
    return probs[:, permutation]


def predict_probs(model_dir: str, config: dict, device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    df_test = load_test_data(config["data_dir"])
    test_dataset = NLIDataset(df_test, tokenizer, config["max_length"], is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting ({model_dir})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    return align_probs_to_canonical(probs, id2label)


def predict_ensemble(model_dirs: list[str], config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Ensembling {len(model_dirs)} models: {model_dirs}")

    per_model_probs = [predict_probs(model_dir, config, device) for model_dir in model_dirs]
    avg_probs = np.mean(per_model_probs, axis=0)
    # argmax index i == Kaggle label i directly, since CANONICAL_LABELS follows Kaggle's own order
    preds = np.argmax(avg_probs, axis=-1)

    df_test = load_test_data(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"id": df_test["id"], "prediction": preds})
    out_path = output_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Ensemble submission saved to: {out_path} ({len(submission)} rows)")
    print(f"Prediction distribution:\n{submission['prediction'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    predict_ensemble(args.model_dirs, load_config(args.config))
```

- [ ] **Step 3: Re-run the label-alignment check**

Same command as Step 1. Expected: `OK` printed.

- [ ] **Step 4: Commit**

```bash
git add src/predict_ensemble.py
git commit -m "feat: add probability-averaging ensemble prediction script"
```

---

### Task 4: Kaggle K-fold training kernel

**Files:**
- Create: `kaggle/watson-kfold-xnli/kernel-metadata.json`
- Create: `kaggle/watson-kfold-xnli/kernel.py`

**Interfaces:**
- Produces (once run on Kaggle, not locally): kernel output `/kaggle/working/outputs/fold_{0..4}/best_model/`, consumed by Task 8's dataset upload step.

Kaggle kernels run standalone — they cannot `from dataset import ...` this repo's `src/` layout — so this vendors the minimal needed code (`NLIDataset`, `compute_metrics`, `make_stratified_folds`) as a single self-contained script, mirroring Task 1's logic and `configs/config.yaml`'s hyperparameters exactly. `report_to="none"` (not `"mlflow"`) because Kaggle kernels have no access to the local MLflow tracking server.

- [ ] **Step 1: Create `kaggle/watson-kfold-xnli/kernel-metadata.json`**

```json
{
  "id": "st4r4x/watson-kfold-xnli",
  "title": "watson-kfold-xnli",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["contradictory-my-dear-watson"],
  "dataset_sources": [],
  "kernel_sources": []
}
```

- [ ] **Step 2: Create `kaggle/watson-kfold-xnli/kernel.py`**

```python
"""Kaggle kernel: 5-fold fine-tune of xlm-roberta-large-xnli on the competition data.
Standalone script — Kaggle kernels don't have this repo's src/ package layout, so the
minimal needed pieces of dataset.py/train.py are vendored here directly.
"""

import os

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

        metrics = trainer.evaluate()
        print(f"Fold {i} val metrics: {metrics}")
        fold_accuracies.append(metrics["eval_accuracy"])

    avg_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\nAverage accuracy across {N_FOLDS} folds: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the script is syntactically valid and the fold-partition assertion holds**

```bash
python3 -m py_compile kaggle/watson-kfold-xnli/kernel.py
python3 - <<'EOF'
import sys
sys.path.insert(0, "kaggle/watson-kfold-xnli")
import pandas as pd
from kernel import make_stratified_folds

df = pd.DataFrame({"label": [0, 1, 2] * 20})
folds = make_stratified_folds(df, n_splits=5, seed=42)
assert len(folds) == 5
print("OK")
EOF
```

Expected: no compile errors, `OK` printed. (This does not run the actual Kaggle training — that requires pushing to Kaggle, done in Task 8.)

- [ ] **Step 4: Commit**

```bash
git add kaggle/watson-kfold-xnli/
git commit -m "feat: add Kaggle kernel for 5-fold xlm-roberta-large-xnli training"
```

---

### Task 5: Kaggle mDeBERTa training kernel

**Files:**
- Create: `kaggle/watson-mdeberta-xnli/kernel-metadata.json`
- Create: `kaggle/watson-mdeberta-xnli/kernel.py`

**Interfaces:**
- Produces (once run on Kaggle): kernel output `/kaggle/working/outputs/best_model/`, consumed by Task 8's dataset upload step.

- [ ] **Step 1: Create `kaggle/watson-mdeberta-xnli/kernel-metadata.json`**

```json
{
  "id": "st4r4x/watson-mdeberta-xnli",
  "title": "watson-mdeberta-xnli",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["contradictory-my-dear-watson"],
  "dataset_sources": [],
  "kernel_sources": []
}
```

- [ ] **Step 2: Create `kaggle/watson-mdeberta-xnli/kernel.py`**

```python
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
```

- [ ] **Step 3: Verify the script is syntactically valid**

```bash
python3 -m py_compile kaggle/watson-mdeberta-xnli/kernel.py
```

Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add kaggle/watson-mdeberta-xnli/
git commit -m "feat: add Kaggle kernel for mDeBERTa-v3-xnli fine-tuning"
```

---

### Task 6: Kaggle ensemble inference kernel

**Files:**
- Create: `kaggle/watson-ensemble-predict/kernel-metadata.json`
- Create: `kaggle/watson-ensemble-predict/kernel.py`

**Interfaces:**
- Consumes: two Kaggle Datasets uploaded in Task 8 — `st4r4x/watson-kfold-xnli-models` (containing `fold_0/best_model/` .. `fold_4/best_model/`) and `st4r4x/watson-mdeberta-xnli-model` (containing `best_model/`).
- Produces: `/kaggle/working/submission.csv`.

Reimplements Task 3's `align_probs_to_canonical` logic (Kaggle kernels can't import `src/predict_ensemble.py` directly) and follows the project's documented dynamic-padding convention for inference (see [CLAUDE.md](../../../CLAUDE.md) submission flow, point 5) since this always runs on Kaggle's shared/CPU-fallback hardware. Also reuses the documented `torch.cuda.get_device_capability()` P100 fallback.

- [ ] **Step 1: Create `kaggle/watson-ensemble-predict/kernel-metadata.json`**

```json
{
  "id": "st4r4x/watson-ensemble-predict",
  "title": "watson-ensemble-predict",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["contradictory-my-dear-watson"],
  "dataset_sources": [
    "st4r4x/watson-kfold-xnli-models",
    "st4r4x/watson-mdeberta-xnli-model"
  ],
  "kernel_sources": []
}
```

- [ ] **Step 2: Create `kaggle/watson-ensemble-predict/kernel.py`**

```python
"""Kaggle kernel: ensemble inference across the 5 K-fold xlm-roberta-large-xnli
checkpoints and the mDeBERTa-v3-xnli checkpoint, averaging softmax probabilities.
Standalone script — see kaggle/watson-kfold-xnli/kernel.py for why.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

COMPETITION_SLUG = "contradictory-my-dear-watson"
TEST_CSV = f"/kaggle/input/competitions/{COMPETITION_SLUG}/test.csv"

KFOLD_DATASET = "/kaggle/input/datasets/st4r4x/watson-kfold-xnli-models"
MDEBERTA_DATASET = "/kaggle/input/datasets/st4r4x/watson-mdeberta-xnli-model"

MODEL_DIRS = [f"{KFOLD_DATASET}/fold_{i}/best_model" for i in range(5)] + [
    f"{MDEBERTA_DATASET}/best_model"
]

MAX_LENGTH = 256
BATCH_SIZE = 8
CANONICAL_LABELS = ["entailment", "neutral", "contradiction"]  # Kaggle's own order


class NLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        # No padding here — dynamic per-batch padding via the collate_fn below is
        # ~4x faster than fixed max_length padding on CPU fallback (P100 not supported).
        return self.tokenizer(
            row["premise"], row["hypothesis"], max_length=self.max_length, truncation=True
        )


def align_probs_to_canonical(probs: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    label_to_idx = {v: k for k, v in id2label.items()}
    missing = set(CANONICAL_LABELS) - set(label_to_idx)
    assert not missing, f"checkpoint id2label is missing labels: {missing} (got {id2label})"
    permutation = [label_to_idx[label] for label in CANONICAL_LABELS]
    return probs[:, permutation]


def predict_probs(model_dir: str, df_test: pd.DataFrame, device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    dataset = NLIDataset(df_test, tokenizer, MAX_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: tokenizer.pad(batch, return_tensors="pt"),
    )

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting ({model_dir})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    return align_probs_to_canonical(probs, id2label)


def main() -> None:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")  # P100 (sm_60) not supported by current torch wheel
    print(f"Using device: {device}")

    df_test = pd.read_csv(TEST_CSV)

    per_model_probs = [predict_probs(model_dir, df_test, device) for model_dir in MODEL_DIRS]
    avg_probs = np.mean(per_model_probs, axis=0)
    preds = np.argmax(avg_probs, axis=-1)  # index i == Kaggle label i (CANONICAL_LABELS order)

    submission = pd.DataFrame({"id": df_test["id"], "prediction": preds})
    submission.to_csv("submission.csv", index=False)
    print(f"Ensemble submission ({len(MODEL_DIRS)} models) saved: {len(submission)} rows")
    print(submission["prediction"].value_counts())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the script is syntactically valid and the alignment logic matches Task 3's**

```bash
python3 -m py_compile kaggle/watson-ensemble-predict/kernel.py
python3 - <<'EOF'
import sys
sys.path.insert(0, "kaggle/watson-ensemble-predict")
import numpy as np
from kernel import align_probs_to_canonical

reversed_id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
probs = np.array([[0.1, 0.2, 0.7]])
aligned = align_probs_to_canonical(probs, reversed_id2label)
assert np.allclose(aligned, [[0.7, 0.2, 0.1]])
print("OK")
EOF
```

Expected: no compile errors, `OK` printed.

- [ ] **Step 4: Commit**

```bash
git add kaggle/watson-ensemble-predict/
git commit -m "feat: add Kaggle ensemble inference kernel"
```

---

### Task 7: Local continued-pretraining on full XNLI

**Files:**
- Create: `src/continued_pretrain.py`
- Create: `configs/config_xnli_pretrained.yaml`

**Interfaces:**
- Consumes: `NLIDataset` from `src/dataset.py`, `build_model_init` and `load_config` from `src/train.py` (Task 1).
- Produces: `outputs/xnli_pretrained/` checkpoint, usable as `model_name` in `configs/config_xnli_pretrained.yaml` for a subsequent normal `python src/train.py --config configs/config_xnli_pretrained.yaml` run.

Runs locally (desktop), not on Kaggle — needs the full `facebook/xnli` corpus (392.7k pairs per language × 15 languages, confirmed via the Hub), which is far too slow to re-download inside a fresh Kaggle kernel session every run. The competition's 15 `lang_abv` codes (verified against `data/train.csv`) exactly match XNLI's 15 per-language configs (`ar, bg, de, el, en, es, fr, hi, ru, sw, th, tr, ur, vi, zh`), so each is loaded via its flat `premise`/`hypothesis`/`label` config rather than the nested `all_languages` config. A `--samples_per_language` flag caps how much of each 392.7k-row split gets used — the full 5.9M-row corpus would take far too long for an exploratory run — and the script logs exactly how much it used out of what's available, since silently training on a small unlabeled fraction of "the full XNLI corpus" would be misleading.

- [ ] **Step 1: Create `src/continued_pretrain.py`**

```python
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
```

- [ ] **Step 2: Create `configs/config_xnli_pretrained.yaml`**

Copy of `configs/config.yaml` with only `model_name` and `output_dir` changed, for the follow-up competition fine-tune once `continued_pretrain.py` has produced a checkpoint:

```yaml
model_name: outputs/xnli_pretrained   # output of src/continued_pretrain.py — same label scheme as config.yaml's xlm-roberta-large-xnli, since it was built with the same label_map/checkpoint_id2label

label_map: {0: 2, 1: 1, 2: 0}
checkpoint_id2label: {0: contradiction, 1: neutral, 2: entailment}

data_dir: data/
output_dir: outputs/xnli_pretrained_finetuned/
language_filter: all
val_split: 0.15
seed: 42

max_length: 256

batch_size: 8
num_epochs: 5
learning_rate: 1.409e-5
warmup_ratio: 0.115
weight_decay: 0.087
gradient_accumulation_steps: 2
bf16: false

metric_for_best_model: accuracy
load_best_model_at_end: true
eval_strategy: epoch
save_strategy: epoch
```

- [ ] **Step 3: Smoke-test with a tiny sample (downloads a small slice per language, ~15 small requests, runs 1 quick training step)**

```bash
python3 src/continued_pretrain.py \
    --config configs/config.yaml \
    --output_dir /tmp/xnli_smoke_test \
    --samples_per_language 5
```

Expected: prints `Loaded <N> XNLI pairs (5/language x 15 languages, out of 392702/language available)`, then `XNLI pretrain — train: ... | val: ...`, trains briefly, ends with `XNLI-continued-pretrained checkpoint saved to: /tmp/xnli_smoke_test`, no traceback. (`n_val = min(5000, len(df)//10)` — with `--samples_per_language 5` that's `75//10=7`, so this exercises the small-dataset edge case too.) Clean up afterward: `gio trash /tmp/xnli_smoke_test`.

- [ ] **Step 4: Commit**

```bash
git add src/continued_pretrain.py configs/config_xnli_pretrained.yaml
git commit -m "feat: add continued pretraining on full XNLI corpus"
```

---

### Task 8: Push kernels, upload models, run the ensemble (operational runbook)

This task has no automated tests — it's driving real Kaggle cloud training (hours of wall-clock time) and uploading real artifacts. Each step's "expected" is what to look for when you check back, not an instant pass/fail.

**Files:** none (CLI operations only)

- [ ] **Step 1: Push the two training kernels**

```bash
kaggle kernels push -p kaggle/watson-kfold-xnli/
kaggle kernels push -p kaggle/watson-mdeberta-xnli/
```

Expected: `Kernel version ... successfully pushed` for each.

- [ ] **Step 2: Select GPU T4×2 for both kernels before they run**

The CLI's `--accelerator` flag exists but its accepted value strings aren't documented in the installed `kaggle` package (`kagglesdk` enum, not exposed). Rather than guess a string and risk silently falling back to the wrong accelerator: open each kernel at `https://www.kaggle.com/code/st4r4x/watson-kfold-xnli` and `https://www.kaggle.com/code/st4r4x/watson-mdeberta-xnli`, open the notebook editor, and under Session options manually select **GPU T4 x2** (not P100 — unsupported by the current torch wheel, per this project's existing convention). Then start the run from the editor. This is a one-time choice per kernel; it's remembered for subsequent pushes.

- [ ] **Step 3: Wait for both kernels to finish, checking status periodically**

```bash
kaggle kernels status st4r4x/watson-kfold-xnli
kaggle kernels status st4r4x/watson-mdeberta-xnli
```

Expected eventually: `"complete"`. If `"error"`, fetch the log with `kaggle kernels output st4r4x/watson-kfold-xnli -p /tmp/kfold_debug` and check `*.log`.

- [ ] **Step 4: Download each kernel's output**

```bash
kaggle kernels output st4r4x/watson-kfold-xnli -p outputs/kfold_from_kaggle/
kaggle kernels output st4r4x/watson-mdeberta-xnli -p outputs/mdeberta_from_kaggle/
```

Expected: `outputs/kfold_from_kaggle/outputs/fold_0/best_model/` .. `fold_4/best_model/` each containing `config.json`, `model.safetensors`, tokenizer files; `outputs/mdeberta_from_kaggle/outputs/best_model/` similarly.

- [ ] **Step 5: Upload each as a Kaggle Dataset — `--dir-mode zip` is required or the subdirectories are silently dropped**

```bash
mkdir -p outputs/kfold_from_kaggle/outputs
cat > outputs/kfold_from_kaggle/outputs/dataset-metadata.json <<'EOF'
{
  "title": "watson-kfold-xnli-models",
  "id": "st4r4x/watson-kfold-xnli-models",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
kaggle datasets create -p outputs/kfold_from_kaggle/outputs/ --dir-mode zip

mkdir -p outputs/mdeberta_from_kaggle/outputs
cat > outputs/mdeberta_from_kaggle/outputs/dataset-metadata.json <<'EOF'
{
  "title": "watson-mdeberta-xnli-model",
  "id": "st4r4x/watson-mdeberta-xnli-model",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
kaggle datasets create -p outputs/mdeberta_from_kaggle/outputs/ --dir-mode zip
```

Expected: `Your private Dataset is being created...` then a URL for each, matching the `dataset_sources` ids already set in `kaggle/watson-ensemble-predict/kernel-metadata.json` (Task 6).

- [ ] **Step 6: Push and run the ensemble inference kernel**

```bash
kaggle kernels push -p kaggle/watson-ensemble-predict/
```

Open `https://www.kaggle.com/code/st4r4x/watson-ensemble-predict`, select GPU T4 x2, run it. Wait for `"complete"` (same polling as Step 3).

- [ ] **Step 7: Download the submission and submit**

```bash
kaggle kernels output st4r4x/watson-ensemble-predict -p outputs/ensemble/
kaggle competitions submit contradictory-my-dear-watson \
    -f outputs/ensemble/submission.csv \
    -m "K-fold xlm-roberta-large-xnli + mDeBERTa-v3-xnli ensemble"
```

Expected: submission accepted, check the public score against the 0.91241 baseline via `kaggle competitions submissions contradictory-my-dear-watson`.
