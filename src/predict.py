"""
Generate submission.csv from a trained model.

Usage:
    python src/predict.py --model_dir outputs/best_model --config configs/config.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from dataset import NLIDataset, load_test_data


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def predict(model_dir: str, config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    df_test = load_test_data(config["data_dir"])
    test_dataset = NLIDataset(df_test, tokenizer, config["max_length"], is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"id": df_test["id"], "prediction": all_preds})
    out_path = output_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path} ({len(submission)} rows)")
    print(f"Prediction distribution:\n{submission['prediction'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/best_model")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    predict(args.model_dir, load_config(args.config))
