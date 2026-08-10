"""Passthrough kernel: copies a submission.csv computed elsewhere (locally, via
src/predict_ensemble.py) into this kernel's output, so it can be submitted to a
kernels-only competition without re-running inference on Kaggle.
"""

import shutil

shutil.copy(
    "/kaggle/input/datasets/st4r4x/watson-ensemble-submission-local/submission.csv",
    "submission.csv",
)
print("Copied submission.csv through.")
