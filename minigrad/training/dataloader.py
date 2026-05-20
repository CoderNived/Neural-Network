"""
training/dataloader.py
──────────────────────
Dataset and DataLoader for mini-batch training.

Design:
  Dataset   — thin wrapper around (X, y) pairs; supports indexing and len
  DataLoader — yields shuffled mini-batches each epoch

Why mini-batches?
  Online SGD (batch_size=1): noisy gradients, cheap per-step, may escape local minima.
  Full-batch GD (batch_size=N): exact gradients, expensive, smooth loss curve.
  Mini-batch: the practical middle ground — enough signal to smooth noise,
  cheap enough to step frequently. Standard in all production frameworks.

Why shuffle?
  Without shuffling, the model sees samples in the same order every epoch.
  It can overfit to the sequence rather than the distribution — particularly
  bad for sorted or structured datasets. Shuffling breaks correlations
  between consecutive batches.
"""

import random


class Dataset:
    """
    Wraps a list of (x, y) pairs.
    x: list of floats (one sample's features)
    y: float or list of floats (target)
    """

    def __init__(self, X, y):
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got {len(X)} vs {len(y)}")
        self.X = list(X)
        self.y = list(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __repr__(self):
        n_features = len(self.X[0]) if self.X else 0
        return f"Dataset(n={len(self)}, features={n_features})"

    @classmethod
    def from_pairs(cls, pairs):
        """Construct from list of (x, y) tuples."""
        X = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        return cls(X, y)

    def split(self, val_fraction=0.2, seed=None):
        """
        Returns (train_dataset, val_dataset).
        Stratification is not implemented — use on shuffled data.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        indices = list(range(len(self)))
        rng.shuffle(indices)
        split_at = int(len(self) * (1 - val_fraction))
        train_idx = indices[:split_at]
        val_idx   = indices[split_at:]

        train = Dataset([self.X[i] for i in train_idx],
                        [self.y[i] for i in train_idx])
        val   = Dataset([self.X[i] for i in val_idx],
                        [self.y[i] for i in val_idx])
        return train, val


class DataLoader:
    """
    Yields mini-batches from a Dataset.

    Each call to __iter__ produces one full pass over the data
    (one epoch), optionally shuffled.

    batch_size=None or batch_size >= len(dataset) → full-batch mode.
    drop_last=True  → discard the final incomplete batch.
    drop_last=False → include it (default, important for small datasets).
    """

    def __init__(self, dataset, batch_size=None, shuffle=True,
                 drop_last=False, seed=None):
        self.dataset    = dataset
        self.batch_size = batch_size or len(dataset)
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self._rng       = random.Random(seed) if seed is not None else random

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)

        batch_X, batch_y = [], []
        for idx in indices:
            x, y = self.dataset[idx]
            batch_X.append(x)
            batch_y.append(y)
            if len(batch_X) == self.batch_size:
                yield batch_X, batch_y
                batch_X, batch_y = [], []

        # Final partial batch
        if batch_X and not self.drop_last:
            yield batch_X, batch_y

    def __len__(self):
        """Number of batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __repr__(self):
        return (f"DataLoader(n={len(self.dataset)}, "
                f"batch_size={self.batch_size}, "
                f"shuffle={self.shuffle}, "
                f"n_batches={len(self)})")