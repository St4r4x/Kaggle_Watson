# Kaggle Watson — Project Instructions

## Context
Kaggle competition: **Contradictory, My Dear Watson** (NLI, 15 languages).
Personal project — repo: `git@github.com-personal:St4r4x/Kaggle_Watson.git`

## Stack
- Python, PyTorch, HuggingFace Transformers
- Model: `microsoft/deberta-v3-base` (encoder-only, fine-tuned for NLI)
- Config driven: all hyperparameters in `configs/config.yaml`

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
```

## Strategy
- Phase 1: English-only (`language_filter: english`, ~6 870 samples)
- Phase 2: full multilingual (`language_filter: all`, ~12 120 samples)
- Next step after Phase 2: try `microsoft/deberta-v3-large` or `xlm-roberta-large`

## Relevant skills
- `/training-check` — review training loop before launching
- `/py-review src/` — review Python code quality
- `/analyze-model` — analyze model architecture choices

## Agents
No custom agents configured for this project yet.
