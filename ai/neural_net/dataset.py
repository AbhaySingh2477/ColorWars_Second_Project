"""
dataset.py — PyTorch Dataset for Color Wars training data.

Wraps the training examples extracted from GameRecords into a PyTorch
Dataset that can be consumed by a DataLoader for batched training.

Data Flow:
──────────
  GameRecord files (JSON)
      ↓
  TrainingPipeline.extract_examples()
      ↓
  (board_state, action_probs, outcome) tuples
      ↓
  TrainingPipeline.encode_examples()
      ↓
  (encoded_board, flat_policy, value) tuples
      ↓
  ColorWarsDataset  ← this module
      ↓
  DataLoader (batched tensors for training)

Design Decisions:
─────────────────
  • Tensors are created in __getitem__ to keep memory usage low
    (raw data stays as Python lists until accessed).
  • Train/val split is random with optional seed for reproducibility.
  • Supports both augmented and non-augmented data.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader


class ColorWarsDataset(Dataset):
    """
    PyTorch Dataset for Color Wars training examples.

    Each example is a tuple:
      (encoded_board, flat_policy, value)

    Where:
      • encoded_board: 5 planes × grid_size × grid_size (float lists)
      • flat_policy: grid_size² probabilities (float list)
      • value: +1.0 or -1.0 (float)

    Usage:
        examples = pipeline.encode_examples(raw_examples, perspective=True)
        dataset = ColorWarsDataset(examples)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for boards, policies, values in loader:
            # boards:   [batch, 5, 6, 6]
            # policies: [batch, 36]
            # values:   [batch, 1]
            log_policy, pred_value = model(boards)
    """

    def __init__(self, examples):
        """
        Args:
            examples: List of (encoded_board, flat_policy, value) tuples.
                      encoded_board: List of 5 planes (grid_size × grid_size).
                      flat_policy:   List of grid_size² float probabilities.
                      value:         Float (+1.0 or -1.0).
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns:
            Tuple (board_tensor, policy_tensor, value_tensor):
              - board_tensor:  [5, grid_size, grid_size] float32
              - policy_tensor: [grid_size²] float32
              - value_tensor:  [1] float32
        """
        board_planes, flat_policy, value = self.examples[idx]

        board_tensor = torch.tensor(board_planes, dtype=torch.float32)
        policy_tensor = torch.tensor(flat_policy, dtype=torch.float32)
        value_tensor = torch.tensor([value], dtype=torch.float32)

        return board_tensor, policy_tensor, value_tensor


def create_data_loaders(examples, batch_size=64, train_split=0.9, seed=None):
    """
    Create train and validation DataLoaders from training examples.

    Args:
        examples:    List of (encoded_board, flat_policy, value) tuples.
        batch_size:  Batch size for both loaders.
        train_split: Fraction of data for training (rest for validation).
        seed:        Random seed for reproducible splits.

    Returns:
        Tuple (train_loader, val_loader):
          - train_loader: DataLoader for training data
          - val_loader:   DataLoader for validation data (may be empty)
    """
    if not examples:
        raise ValueError("No training examples provided")

    # Shuffle and split
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]

    train_dataset = ColorWarsDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = None
    if val_examples:
        val_dataset = ColorWarsDataset(val_examples)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, val_loader
