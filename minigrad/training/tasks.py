"""
training/tasks.py
──────────────────
Canonical datasets and recommended network configurations for each task.

Each task returns a TaskSpec with:
  dataset         : Dataset (full)
  recommended_mlp : dict of kwargs for MLP constructor
  loss_fn         : which loss to use
  metric_threshold: accuracy boundary for binary tasks
  description     : what the task tests

Design note: XOR is the canonical non-linearly-separable problem.
A single neuron cannot learn XOR — no hyperplane separates the classes.
It requires at least one hidden layer (two linear regions composed).
AND and OR are linearly separable — a single neuron can learn them.
Regression tests that the network can fit a continuous function.
"""

from training.dataloader import Dataset
from losses.losses import mse, bce


class TaskSpec:
    def __init__(self, name, dataset, val_dataset=None,
                 recommended_arch=None, loss_fn=None,
                 metric_threshold=0.5, description=""):
        self.name              = name
        self.dataset           = dataset
        self.val_dataset       = val_dataset
        self.recommended_arch  = recommended_arch or {}
        self.loss_fn           = loss_fn or mse
        self.metric_threshold  = metric_threshold
        self.description       = description

    def __repr__(self):
        return (f"TaskSpec(name={self.name!r}, "
                f"n={len(self.dataset)}, "
                f"loss={self.loss_fn.__name__})")


# ─────────────────────────────────────────
# BINARY LOGIC TASKS
# ─────────────────────────────────────────

def and_task():
    """
    AND: output 1 only when both inputs are 1.
    Linearly separable. Single neuron + sigmoid + MSE converges easily.
    """
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0,        0.0,        0.0,        1.0       ]
    return TaskSpec(
        name             = 'AND',
        dataset          = Dataset(X, y),
        recommended_arch = {
            'layer_sizes': [2, 1],
            'activations': ['sigmoid'],
        },
        loss_fn          = mse,
        metric_threshold = 0.5,
        description      = (
            "Linearly separable. Tests whether a single sigmoid neuron "
            "can find the separating hyperplane. Should converge reliably."
        ),
    )


def or_task():
    """
    OR: output 1 when at least one input is 1.
    Linearly separable. Easier than AND (3 out of 4 positive).
    """
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0,        1.0,        1.0,        1.0       ]
    return TaskSpec(
        name             = 'OR',
        dataset          = Dataset(X, y),
        recommended_arch = {
            'layer_sizes': [2, 1],
            'activations': ['sigmoid'],
        },
        loss_fn          = mse,
        metric_threshold = 0.5,
        description      = (
            "Linearly separable. Faster to converge than AND. "
            "Good sanity check before XOR."
        ),
    )


def xor_task():
    """
    XOR: output 1 when inputs differ.
    NOT linearly separable. A single neuron will fail — its loss
    will plateau above 0.25 (chance level for this encoding).
    Requires hidden layer to compose two half-planes.

    Canonical architecture: [2, 4, 1] with tanh hidden, sigmoid output.
    """
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0,        1.0,        1.0,        0.0       ]
    return TaskSpec(
        name             = 'XOR',
        dataset          = Dataset(X, y),
        recommended_arch = {
            'layer_sizes': [2, 4, 1],
            'activations': ['tanh', 'sigmoid'],
        },
        loss_fn          = mse,
        metric_threshold = 0.5,
        description      = (
            "NOT linearly separable. Tests MLP expressiveness. "
            "Single neuron will plateau around loss=0.25 — this is expected "
            "and is itself a diagnostic: it confirms the network architecture "
            "is the binding constraint, not the optimizer."
        ),
    )


def xor_neuron_task():
    """Same XOR data but with single-neuron architecture — demonstrates failure."""
    spec = xor_task()
    spec.name             = 'XOR (single neuron — expected failure)'
    spec.recommended_arch = {
        'layer_sizes': [2, 1],
        'activations': ['sigmoid'],
    }
    spec.description = (
        "Intentional failure case: single neuron cannot separate XOR. "
        "Loss will plateau above 0.25. This is not a bug — it's the "
        "fundamental limitation of linear classifiers."
    )
    return spec


# ─────────────────────────────────────────
# REGRESSION TASK
# ─────────────────────────────────────────

def regression_task(n_points=20, noise=0.05, seed=42):
    """
    Regression: learn f(x) = sin(2πx) on [0, 1].

    Tests:
    - Can the MLP fit a continuous nonlinear function?
    - Does MSE loss drive predictions toward the true values?
    - Does the network generalize (low val loss)?

    Uses a train/val split to check generalization.
    """
    import math
    import random
    rng = random.Random(seed)

    # Training points
    X_train, y_train = [], []
    for _ in range(n_points):
        x   = rng.uniform(0, 1)
        y   = math.sin(2 * math.pi * x) + rng.gauss(0, noise)
        X_train.append([x])
        y_train.append(y)

    # Validation points (different seed)
    rng2 = random.Random(seed + 1)
    X_val, y_val = [], []
    for _ in range(10):
        x  = rng2.uniform(0, 1)
        y  = math.sin(2 * math.pi * x) + rng2.gauss(0, noise)
        X_val.append([x])
        y_val.append(y)

    return TaskSpec(
        name             = 'Regression (sin 2πx)',
        dataset          = Dataset(X_train, y_train),
        val_dataset      = Dataset(X_val, y_val),
        recommended_arch = {
            'layer_sizes': [1, 8, 8, 1],
            'activations': ['tanh', 'tanh', 'linear'],
        },
        loss_fn          = mse,
        metric_threshold = None,   # not a classification task
        description      = (
            "Nonlinear regression. Tests whether MLP can fit a continuous "
            "function with noise. Output layer is linear (unbounded). "
            "Watch val loss: if train loss decreases but val loss grows, "
            "the network is overfitting."
        ),
    )


# ─────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────

ALL_TASKS = {
    'and':        and_task,
    'or':         or_task,
    'xor':        xor_task,
    'xor_neuron': xor_neuron_task,
    'regression': regression_task,
}

def get_task(name: str) -> TaskSpec:
    if name not in ALL_TASKS:
        raise ValueError(f"Unknown task {name!r}. Available: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[name]()