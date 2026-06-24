"""
trainer.py — Training loop for the Color Wars neural network.

This module handles the complete training workflow:
  1. Load game records from disk
  2. Extract and encode training examples
  3. Create PyTorch DataLoaders
  4. Train the model with combined policy + value loss
  5. Evaluate on validation data
  6. Save checkpoints

Loss Function (AlphaZero):
──────────────────────────
  L = (z - v)² - π·log(p) + c·‖θ‖²

  Where:
    z = actual game outcome (+1 or -1)
    v = predicted value from value head
    π = MCTS policy (target probabilities)
    p = predicted policy from policy head
    c = L2 regularization coefficient (weight_decay in Adam)

  • Value loss:  MSE between predicted and actual outcome
  • Policy loss: Cross-entropy between MCTS policy and predicted policy
  • L2 regularization is handled by Adam's weight_decay parameter

Training Metrics:
─────────────────
  • total_loss:     Combined policy + value loss
  • policy_loss:    Cross-entropy on move probabilities
  • value_loss:     MSE on game outcome prediction
  • policy_accuracy: How often the top-1 predicted move matches MCTS's top move
  • value_mae:      Mean absolute error on value predictions
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from ai.neural_net import config
from ai.neural_net.model import ColorWarsNet, get_device
from ai.neural_net.dataset import create_data_loaders
from ai.self_play.training_pipeline import TrainingPipeline


class Trainer:
    """
    Trains the ColorWarsNet model on MCTS self-play data.

    Usage:
        trainer = Trainer()
        trainer.load_data("data/mcts_self_play", augment=True)
        metrics = trainer.train(n_epochs=10)
        trainer.save_checkpoint("ai/neural_net/checkpoints/model_v1.pt")
    """

    def __init__(self, model=None, device=None, lr=config.LEARNING_RATE,
                 weight_decay=config.WEIGHT_DECAY, batch_size=config.BATCH_SIZE):
        """
        Args:
            model:        ColorWarsNet instance. If None, creates a new one.
            device:       Torch device. If None, auto-detects.
            lr:           Learning rate for Adam optimizer.
            weight_decay: L2 regularization coefficient.
            batch_size:   Training batch size.
        """
        self.device = device or get_device()
        self.model = model or ColorWarsNet()
        self.model.to(self.device)
        self.batch_size = batch_size

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Loss functions
        # Policy: NLLLoss expects log-probabilities (model outputs log_softmax)
        # We use a custom cross-entropy since our targets are probability
        # distributions, not class indices.
        # Value: MSE loss
        self.value_criterion = nn.MSELoss()

        # Training state
        self.train_loader = None
        self.val_loader = None
        self.current_epoch = 0
        self.training_history = []

    # ─── Data Loading ────────────────────────────────────────────────

    def load_data(self, data_dir, augment=config.AUGMENT_DATA,
                  policy_only=config.POLICY_ONLY,
                  perspective=config.USE_PERSPECTIVE,
                  max_games=None, seed=42):
        """
        Load game records and prepare DataLoaders.

        Args:
            data_dir:    Directory containing game JSON files.
            augment:     Apply 8-fold symmetry augmentation.
            policy_only: Skip examples without MCTS policy data.
            perspective: Encode from current player's perspective.
            max_games:   Max number of games to load.
            seed:        Random seed for train/val split.

        Returns:
            Tuple (n_train, n_val) — number of examples in each split.
        """
        pipeline = TrainingPipeline(grid_size=config.GRID_SIZE)

        # Load game records
        print(f"  Loading games from: {data_dir}")
        records = pipeline.load_games(data_dir, max_games=max_games)

        # Extract raw training examples
        raw_examples = pipeline.extract_examples(
            records, augment=augment, policy_only=policy_only,
        )
        print(f"  Raw examples: {len(raw_examples)}")

        # Encode for neural network
        encoded_examples = pipeline.encode_examples(
            raw_examples, perspective=perspective,
        )
        print(f"  Encoded examples: {len(encoded_examples)}")

        # Create DataLoaders
        self.train_loader, self.val_loader = create_data_loaders(
            encoded_examples,
            batch_size=self.batch_size,
            train_split=config.TRAIN_VAL_SPLIT,
            seed=seed,
        )

        n_train = len(self.train_loader.dataset)
        n_val = len(self.val_loader.dataset) if self.val_loader else 0

        print(f"  Train examples: {n_train}")
        print(f"  Val examples:   {n_val}")

        return n_train, n_val

    def load_from_examples(self, encoded_examples, seed=42):
        """
        Create DataLoaders from pre-encoded examples.

        Args:
            encoded_examples: List of (encoded_board, flat_policy, value) tuples.
            seed:             Random seed for train/val split.

        Returns:
            Tuple (n_train, n_val).
        """
        self.train_loader, self.val_loader = create_data_loaders(
            encoded_examples,
            batch_size=self.batch_size,
            train_split=config.TRAIN_VAL_SPLIT,
            seed=seed,
        )

        n_train = len(self.train_loader.dataset)
        n_val = len(self.val_loader.dataset) if self.val_loader else 0
        return n_train, n_val

    # ─── Training ────────────────────────────────────────────────────

    def _policy_loss(self, log_policy, target_policy):
        """
        Cross-entropy loss between predicted and target policy distributions.

        Unlike standard cross-entropy (which expects class indices), our
        targets are full probability distributions from MCTS. So we compute:
            L = -Σ π(a) · log(p(a))

        Args:
            log_policy:    [batch, 36] — model output (log-probabilities)
            target_policy: [batch, 36] — MCTS policy (probabilities)

        Returns:
            Scalar loss tensor.
        """
        return -torch.sum(target_policy * log_policy, dim=1).mean()

    def train(self, n_epochs=config.NUM_EPOCHS, verbose=True):
        """
        Train the model for n_epochs.

        Args:
            n_epochs: Number of training epochs.
            verbose:  Print progress per epoch.

        Returns:
            List of epoch metric dictionaries.
        """
        if self.train_loader is None:
            raise RuntimeError("No training data loaded. Call load_data() first.")

        self.model.train()
        all_metrics = []

        print(f"\n{'═' * 55}")
        print(f"  TRAINING — {n_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"{'═' * 55}\n")

        for epoch in range(1, n_epochs + 1):
            self.current_epoch += 1
            epoch_start = time.time()

            # ── Training Pass ──
            train_metrics = self._train_epoch()

            # ── Validation Pass ──
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._evaluate(self.val_loader)

            elapsed = time.time() - epoch_start

            # ── Log ──
            metrics = {
                'epoch': self.current_epoch,
                'train': train_metrics,
                'val': val_metrics,
                'elapsed': elapsed,
            }
            all_metrics.append(metrics)
            self.training_history.append(metrics)

            if verbose:
                val_str = ""
                if val_metrics:
                    val_str = (f" | val_loss={val_metrics['total_loss']:.4f}"
                               f" val_acc={val_metrics['policy_accuracy']:.1f}%")
                print(
                    f"  Epoch {self.current_epoch:3d} | "
                    f"loss={train_metrics['total_loss']:.4f} "
                    f"(p={train_metrics['policy_loss']:.4f} "
                    f"v={train_metrics['value_loss']:.4f}) | "
                    f"acc={train_metrics['policy_accuracy']:.1f}%"
                    f"{val_str} | "
                    f"{elapsed:.1f}s"
                )

        print(f"\n  Training complete. Final loss: "
              f"{all_metrics[-1]['train']['total_loss']:.4f}\n")

        return all_metrics

    def _train_epoch(self):
        """
        Run one training epoch.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_top1 = 0
        total_samples = 0

        for boards, policies, values in self.train_loader:
            boards = boards.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)

            # Forward pass
            log_policy, pred_value = self.model(boards)

            # Compute losses
            p_loss = self._policy_loss(log_policy, policies)
            v_loss = self.value_criterion(pred_value, values)
            loss = (config.POLICY_LOSS_WEIGHT * p_loss +
                    config.VALUE_LOSS_WEIGHT * v_loss)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            batch_size = boards.size(0)
            total_loss += loss.item() * batch_size
            total_policy_loss += p_loss.item() * batch_size
            total_value_loss += v_loss.item() * batch_size
            total_samples += batch_size

            # Policy accuracy: does top-1 prediction match MCTS top-1?
            pred_moves = torch.argmax(log_policy, dim=1)
            target_moves = torch.argmax(policies, dim=1)
            correct_top1 += (pred_moves == target_moves).sum().item()

        n = total_samples
        return {
            'total_loss': total_loss / n,
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'policy_accuracy': (correct_top1 / n) * 100,
            'n_samples': n,
        }

    # ─── Evaluation ──────────────────────────────────────────────────

    def _evaluate(self, data_loader):
        """
        Evaluate the model on a DataLoader (no gradient updates).

        Args:
            data_loader: PyTorch DataLoader.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_top1 = 0
        total_value_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for boards, policies, values in data_loader:
                boards = boards.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                log_policy, pred_value = self.model(boards)

                p_loss = self._policy_loss(log_policy, policies)
                v_loss = self.value_criterion(pred_value, values)
                loss = (config.POLICY_LOSS_WEIGHT * p_loss +
                        config.VALUE_LOSS_WEIGHT * v_loss)

                batch_size = boards.size(0)
                total_loss += loss.item() * batch_size
                total_policy_loss += p_loss.item() * batch_size
                total_value_loss += v_loss.item() * batch_size
                total_samples += batch_size

                pred_moves = torch.argmax(log_policy, dim=1)
                target_moves = torch.argmax(policies, dim=1)
                correct_top1 += (pred_moves == target_moves).sum().item()

                total_value_error += torch.abs(
                    pred_value - values
                ).sum().item()

        n = total_samples
        return {
            'total_loss': total_loss / n,
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'policy_accuracy': (correct_top1 / n) * 100,
            'value_mae': total_value_error / n,
            'n_samples': n,
        }

    # ─── Checkpoint ──────────────────────────────────────────────────

    def save_checkpoint(self, filepath=None):
        """
        Save a training checkpoint.

        Args:
            filepath: Path to save. If None, uses default naming.

        Returns:
            The filepath where the checkpoint was saved.
        """
        if filepath is None:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            filepath = os.path.join(
                config.CHECKPOINT_DIR,
                f"{config.CHECKPOINT_PREFIX}_epoch{self.current_epoch:03d}.pt"
            )

        self.model.save_checkpoint(
            filepath,
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict(),
            loss=self.training_history[-1]['train']['total_loss']
            if self.training_history else None,
            metadata={
                'training_history': self.training_history,
            },
        )
        print(f"  💾 Checkpoint saved: {filepath}")
        return filepath

    def load_checkpoint(self, filepath):
        """
        Load a training checkpoint and resume training.

        Args:
            filepath: Path to the checkpoint file.
        """
        self.model, checkpoint = ColorWarsNet.load_checkpoint(
            filepath, device=self.device,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']

        if 'metadata' in checkpoint and 'training_history' in checkpoint['metadata']:
            self.training_history = checkpoint['metadata']['training_history']

        print(f"  ✅ Loaded checkpoint: {filepath} (epoch {self.current_epoch})")
