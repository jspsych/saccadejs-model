# The Keras backend MUST be chosen before ANY import below, because
# `et_util.custom_layers` does `import keras` on its first line and keras locks in
# its backend at import time. Python runs this package __init__ before the body of
# any submodule, so `python -m et_util.train` executes these lines *before*
# train.py's own setdefault -- any setdefault placed later is dead code, and keras
# silently falls back to its default backend (tensorflow, when TF is installed).
# That bug ran training on tensorflow-cpu at 109 s/step instead of jax-TPU at
# 0.3 s/step. An explicitly exported KERAS_BACKEND still wins: setdefault is a
# no-op when the variable is already set (analysis scripts that want the
# tensorflow backend for eval-only work rely on this).
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

__version__ = "0.3.0"

# Only Keras-registration imports belong below: everything here runs on every
# `python -m et_util.train`. Import anything else from its submodule directly.

# Import custom layers so they get registered at import time
from et_util.custom_layers import (
    SimpleTimeDistributed,
    MaskedWeightedRidgeRegressionLayer,
    MaskInspectorLayer,
)

# Import custom loss so it gets registered at import time
from et_util.custom_loss import normalized_weighted_euc_dist
