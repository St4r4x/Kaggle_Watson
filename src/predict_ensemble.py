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
