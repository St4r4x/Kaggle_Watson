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
