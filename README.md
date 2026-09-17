# Contradictory, My Dear Watson

[Kaggle competition](https://www.kaggle.com/competitions/contradictory-my-dear-watson): Natural Language Inference (NLI) on 15 languages.
Given a **premise** and a **hypothesis**, classify their relationship:

| Label | Meaning |
|-------|---------|
| 0 | Entailment |
| 1 | Neutral |
| 2 | Contradiction |

## Dataset

- **Train**: 12,120 samples — 15 languages (English ~57%, + Arabic, French, Chinese, etc.)
- **Test**: 5,195 samples

## Results

**Public leaderboard score: 0.92858 — rank 9/53.** Ranks 1-7 sit at a suspicious
1.00000 (this is a beginner "Getting Started" competition with no leakage
protection, so a perfect score is almost certainly test-label lookup rather
than a model); ours is the best score we could find backed by an actual
trained ensemble.

| Model | Val accuracy | Kaggle public score |
|---|---|---|
| BERT-base-multi (baseline notebook) | ~59% | — |
| DeBERTa-v3-base, English only | 85.5% | — |
| DeBERTa-v3-base, 15 languages | 78.0% | — |
| xlm-roberta-large (raw), 15 languages | 81.7% | — |
| xlm-roberta-large-xnli, 15 languages | 92.24% | 0.91241 |
| **5-fold xlm-roberta-large-xnli + mDeBERTa-v3-xnli ensemble** | 92.32% (fold avg) | **0.92858** |

Full write-up of the ensembling technique: [docs/ensemble-techniques.md](docs/ensemble-techniques.md).

## Approach

Started from a KerasNLP baseline notebook, then iterated on the encoder rather than
architecture novelty:

1. **DeBERTa-v3-base** fine-tuned from raw pretraining — first working pipeline.
2. Switched to **`joeddav/xlm-roberta-large-xnli`**, already fine-tuned on MultiNLI+XNLI —
   a big head start over training NLI from scratch, +10pt val accuracy.
3. **5-fold ensemble**: re-trained the same model on 5 stratified folds, plus a second,
   smaller architecture (`mDeBERTa-v3-xnli`) for error diversity, combined by averaging
   softmax probabilities across all 6 models.

Other things in the pipeline: stratified train/val split, AdamW + warmup + weight decay
(tuned via Optuna), early stopping, fp32 training (DeBERTa-family attention underflows
in bf16), dynamic-length tokenization tuned per language script.

## Project structure

```
├── data/                  # CSVs (git-ignored)
├── notebooks/             # Exploratory notebook (KerasNLP baseline)
├── src/
│   ├── dataset.py             # PyTorch Dataset + data loading
│   ├── train.py               # HuggingFace Trainer pipeline (single split, --kfold, --hpo, --rdrop)
│   ├── predict.py             # Generate submission.csv from one model
│   ├── predict_ensemble.py    # Average softmax probs across multiple model dirs
│   └── continued_pretrain.py  # Continued pretraining on the full XNLI corpus
├── configs/                # config.yaml + per-model variants
├── kaggle/                 # Kernel scripts pushed to Kaggle (training + ensemble inference)
├── docs/                   # Ensemble write-up, design/plan notes
├── outputs/                # Checkpoints + submissions (git-ignored)
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Train a single model (config controls english-only vs. all languages)
python src/train.py --config configs/config.yaml

# Train a 5-fold ensemble instead
python src/train.py --config configs/config.yaml --kfold 5

# Generate a submission from one model
python src/predict.py --model_dir outputs/best_model --config configs/config.yaml

# Generate a submission by averaging several models (e.g. the 5 folds + mDeBERTa)
python src/predict_ensemble.py --model_dirs outputs/fold_0/best_model outputs/fold_1/best_model ... --config configs/config.yaml
```

This competition is **kernels-only** (no direct CSV upload) — see [CLAUDE.md](CLAUDE.md#submitting-to-kaggle)
for the local-train / Kaggle-shim-kernel submission flow used to reach the scores above.
