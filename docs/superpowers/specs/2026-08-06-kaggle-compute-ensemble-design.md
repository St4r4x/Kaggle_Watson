# Kaggle compute offload + ensembling — design

## Context

Current best result: `xlm-roberta-large-xnli` fine-tuned locally, 92.24% val accuracy,
0.91241 Kaggle public score (see [CLAUDE.md](../../../CLAUDE.md) Results table).
An R-Drop regularization variant was tried on the desktop machine and underperformed
(90.92% acc, same as a plain 5-epoch run, worse than the 92.24% reference).

Goal: push the score higher by offloading new experiments to Kaggle's free GPU
kernels (kernels-only competition, no direct CSV upload) while keeping the local
desktop free for a longer-running experiment.

## Architecture

Two Kaggle training kernels run in parallel while the desktop runs a continued-pretraining
experiment locally. Each produces a model checkpoint; a final ensemble inference kernel
combines all of them by averaging softmax probabilities.

```
Kaggle kernel 1 (K-fold xlm-roberta-large-xnli) ──┐
Kaggle kernel 2 (mDeBERTa-v3-xnli fine-tune)     ──┼──► ensemble inference kernel ──► submission.csv
Desktop local (continued pretraining on full XNLI) ┘        (avg softmax probs)
```

## Components

1. **`src/train.py`** — add a `--kfold N` flag. Reuses the existing `model_init`/`Trainer`
   setup; loops over a `StratifiedKFold(n_splits=N)` on the language-filtered dataframe
   instead of the current single `train_test_split`; saves each fold's best model to
   `outputs/fold_{i}/best_model`. No change to single-split behavior when `--kfold` is
   omitted.

2. **`configs/config_mdeberta.yaml`** — new config, copy of `config.yaml` with
   `model_name: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. This checkpoint's native
   `id2label` order must be checked against Kaggle's `0=entailment,1=neutral,2=contradiction`
   at implementation time and `label_map`/`checkpoint_id2label` adjusted accordingly —
   do not assume it matches `joeddav/xlm-roberta-large-xnli`'s order.

3. **Two Kaggle training kernels** (`kernel-metadata.json` × 2, pushed via `kaggle kernels push`):
   - `watson-kfold-xnli`: `train.py --config configs/config.yaml --kfold 5`
   - `watson-mdeberta-xnli`: `train.py --config configs/config_mdeberta.yaml`
   Both reference the competition as a `competition_sources` (direct access to
   `/kaggle/input/competitions/<comp-slug>/train.csv`, no separate dataset upload needed
   for input data). GPU accelerator: **T4×2**, not P100 (P100/sm_60 not supported by the
   current torch wheel, per existing project convention in `predict.py`).

4. **`src/continued_pretrain.py`** (run locally on the desktop, not on Kaggle — no time
   pressure there and it needs the full ~392k-row HF `xnli` dataset which is slow to
   re-download per kernel run). Continues fine-tuning the current best checkpoint on the
   full multilingual XNLI dataset before the existing competition-data fine-tune step.
   Output: `outputs/xnli_pretrained/`, usable as `model_name` in a `train.py` config.

5. **`src/predict_ensemble.py`** — new inference script. Takes a list of model directories,
   runs each on the (dynamically padded) test set, averages softmax probabilities across
   all models, argmaxes, writes `submission.csv`. Pushed as the final Kaggle inference
   kernel with one `dataset_sources` entry per uploaded trained model (`kaggle datasets
   create` for each of: 5 fold checkpoints, the mDeBERTa checkpoint, and — if it improves
   the local val score — the XNLI-continued-pretrain model).

## Data flow

```
train.csv (competition input)
  ├─ K-fold split (Kaggle kernel 1)         → 5× best_model/
  ├─ standard split (Kaggle kernel 2)       → 1× best_model/  (mDeBERTa)
  └─ XNLI-pretrained → competition fine-tune (desktop, local) → 1× best_model/
       │
       ▼ (each best_model/ uploaded as a Kaggle Dataset)
ensemble inference kernel → averaged probs → submission.csv → kaggle competitions submit
```

## Robustness / checks

- K-fold: assert fold indices don't overlap and their union covers 100% of the dataset
  (minimal `assert`-based check in `train.py`, not a test framework).
- Ensemble: assert every loaded model's remapped label order matches Kaggle's canonical
  order (0=entailment,1=neutral,2=contradiction) before averaging — silently mixing a
  model with a different native label order would corrupt the ensemble without erroring.
- Reuse the existing GPU-capability fallback from `predict.py` (P100 sm_60 → CPU fallback)
  in the new ensemble inference kernel.

## Out of scope

- Pseudo-labeling on the test set (considered, deferred — higher risk of confirmation
  bias, revisit only if the above doesn't move the score).
- Parallelizing K-fold across 5 separate kernels — one kernel training all 5 folds
  sequentially is simpler to push/monitor and fits within a single Kaggle GPU session;
  revisit only if session time limits become a problem.
