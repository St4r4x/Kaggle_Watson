# Contradictory, My Dear Watson

Kaggle competition: Natural Language Inference (NLI) on 15 languages.
Given a **premise** and a **hypothesis**, classify their relationship:

| Label | Meaning |
|-------|---------|
| 0 | Entailment |
| 1 | Neutral |
| 2 | Contradiction |

## Dataset

- **Train**: 12,120 samples — 15 languages (English ~57%, + Arabic, French, Chinese, etc.)
- **Test**: 5,195 samples

## Approach

Fine-tuning **DeBERTa-v3-base** (Microsoft) — strong encoder for NLI tasks.
Phase 1: English-only training. Phase 2: full multilingual.

Improvements over the baseline KerasNLP notebook:
- DeBERTa-v3 instead of BERT-base-multilingual
- Stratified train/val split
- AdamW + warmup + cosine decay
- Early stopping (patience=2)
- Mixed precision (fp16)

## Project structure

```
├── data/                  # CSVs (git-ignored)
├── notebooks/             # Exploratory notebook (KerasNLP baseline)
├── src/
│   ├── dataset.py         # PyTorch Dataset + data loading
│   ├── train.py           # HuggingFace Trainer pipeline
│   └── predict.py         # Generate submission.csv
├── configs/
│   └── config.yaml        # All hyperparameters
├── outputs/               # Checkpoints + submissions (git-ignored)
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Train (English only)
python src/train.py --config configs/config.yaml

# Generate submission
python src/predict.py --model_dir outputs/best_model --config configs/config.yaml
```

## Results

| Model | Filter | Val Accuracy |
|-------|--------|-------------|
| BERT-base-multi (baseline) | All languages | ~59% |
| DeBERTa-v3-base | English only | TBD |
| DeBERTa-v3-base | All languages | TBD |
