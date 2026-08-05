# Kaggle Watson — Project Instructions

## Context
Kaggle competition: **Contradictory, My Dear Watson** (NLI, 15 languages).
Personal project — repo: `git@github.com-personal:St4r4x/Kaggle_Watson.git`

## Stack
- Python, PyTorch, HuggingFace Transformers
- Model: `joeddav/xlm-roberta-large-xnli` (already fine-tuned on MultiNLI+XNLI — big head start over the raw pretrained checkpoint)
- Config driven: all hyperparameters in `configs/config.yaml`
- Its native label order is `0=contradiction,1=neutral,2=entailment` — the reverse of Kaggle's. `label_map`/`checkpoint_id2label` in `config.yaml` handle the remap on train and predict.
- Train in fp32 only — `torch_dtype=torch.float32` explicit in `from_pretrained` (checkpoints silently load fp16 otherwise), and `bf16: false` (DeBERTa-family attention underflows in bf16; not verified needed for XLM-R but left off for safety)

## Project structure
```
data/          CSVs — git-ignored, extracted from contradictory-my-dear-watson.zip
notebooks/     Exploratory notebooks (KerasNLP baseline)
src/           Training code (dataset.py, train.py, predict.py)
configs/       config.yaml
outputs/       Checkpoints + submissions — git-ignored
```

## Running
```bash
# Train (English only by default)
python src/train.py --config configs/config.yaml

# Generate submission
python src/predict.py --model_dir outputs/best_model --config configs/config.yaml

# Optuna hyperparameter sweep (Trainer.hyperparameter_search)
python src/train.py --config configs/config.yaml --hpo --n-trials 8
```

## Strategy
- Phase 1: English-only (`language_filter: english`, ~6 870 samples) — done, 85.5% acc with deberta-v3-base
- Phase 2: full multilingual (`language_filter: all`, ~12 120 samples) — done, see Results below
- `max_length: 256` for multilingual (Thai/Urdu/Greek/Hindi tokenize much denser than English — 128 truncated Thai on average)

## Results
| Config | Val accuracy | Kaggle public score |
|---|---|---|
| deberta-v3-base, English only | 85.5% | — |
| deberta-v3-base, 15 languages | 78.0% | — |
| xlm-roberta-large (raw), 15 languages | 81.7% | — |
| **xlm-roberta-large-xnli, 15 languages** | **92.24%** | **0.91241** |

Winning hyperparameters (Optuna, n_trials=8): `learning_rate: 1.409e-5`, `warmup_ratio: 0.115`, `weight_decay: 0.087`.

## Submitting to Kaggle
This competition is **kernels-only** (`is_kernels_submissions_only`) — no direct CSV upload. Flow:
1. Train locally/remotely, then upload `outputs/best_model/` as a Kaggle Dataset: `kaggle datasets create -p <model_dir>/` (needs `dataset-metadata.json` with `id`/`title`/`licenses`)
2. Push an inference-only kernel referencing that dataset + the competition as sources (`kernel-metadata.json` → `dataset_sources`, `competition_sources`)
3. Inside the kernel, real mount paths are `/kaggle/input/datasets/<owner>/<slug>/` and `/kaggle/input/competitions/<comp-slug>/` — NOT the flat `/kaggle/input/<slug>/` from older docs
4. `enable_gpu: true` but add a `torch.cuda.get_device_capability()` check with CPU fallback — Kaggle's free-tier P100 (sm_60) isn't supported by current PyTorch wheels; will error otherwise
5. Use dynamic padding (`tokenizer.pad(..., padding=True)`) not fixed `max_length` padding for inference — ~4x faster on CPU fallback
6. Submit with the classic CLI, not the MCP server (`mcp__kaggle__*` write/status endpoints return `Unauthenticated` even though read endpoints work): `kaggle competitions submit <comp> -k <owner>/<kernel-slug> -v <version> -f submission.csv -m "..."`
7. Auth: `kaggle` CLI 2.2.4+ reads `~/.kaggle/access_token` automatically (newer token format, not the classic `kaggle.json` username+key)

## Relevant skills
- `/training-check` — review training loop before launching
- `/py-review src/` — review Python code quality
- `/analyze-model` — analyze model architecture choices

## Agents
No custom agents configured for this project yet.
