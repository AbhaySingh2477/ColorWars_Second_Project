"""
config.py — Hyperparameters and model configuration for Phase 4.

This module centralizes all tunable parameters for the neural network,
training process, and data pipeline. Adjusting these values here
automatically propagates through the entire system.

Sizing Rationale (6×6 board):
─────────────────────────────
  AlphaZero uses 20 ResBlocks / 256 filters for 19×19 Go (361 cells).
  Our 6×6 board has only 36 cells — about 10× smaller.
  We scale down accordingly:
    • 4 ResBlocks (vs 20)     — enough depth for 6×6 patterns
    • 64 filters (vs 256)     — enough capacity without overfitting
    • ~50K parameters total   — trains in seconds per epoch on M3

  This can be scaled up later if the model underfits.
"""

# ─── Board Configuration ────────────────────────────────────────────
GRID_SIZE = 6
NUM_INPUT_PLANES = 5      # P1 own, P1 dots, P2 own, P2 dots, current player
POLICY_SIZE = GRID_SIZE * GRID_SIZE  # 36 possible moves

# ─── Model Architecture ────────────────────────────────────────────
NUM_RES_BLOCKS = 4         # Number of residual blocks in the tower
NUM_FILTERS = 64           # Channels in convolutional layers
KERNEL_SIZE = 3            # Kernel size for conv layers
PADDING = 1                # Padding to preserve spatial dimensions

# Policy head
POLICY_FILTERS = 2         # 1×1 conv filters before the FC layer

# Value head
VALUE_FILTERS = 1          # 1×1 conv filters before the FC layers
VALUE_HIDDEN = 64          # Hidden units in the value head FC layer

# ─── Training Hyperparameters ───────────────────────────────────────
LEARNING_RATE = 0.001      # Adam optimizer learning rate
WEIGHT_DECAY = 1e-4        # L2 regularization coefficient
BATCH_SIZE = 64            # Training batch size
NUM_EPOCHS = 10            # Epochs per training cycle
TRAIN_VAL_SPLIT = 0.9      # 90% train, 10% validation

# ─── Loss Weights ──────────────────────────────────────────────────
POLICY_LOSS_WEIGHT = 1.0   # Weight for cross-entropy policy loss
VALUE_LOSS_WEIGHT = 1.0    # Weight for MSE value loss

# ─── Data Pipeline ─────────────────────────────────────────────────
AUGMENT_DATA = True        # Apply 8-fold symmetry augmentation
POLICY_ONLY = True         # Skip examples without MCTS policy data
USE_PERSPECTIVE = True     # Encode board from current player's perspective

# ─── Checkpoint ────────────────────────────────────────────────────
CHECKPOINT_DIR = "ai/neural_net/checkpoints"
CHECKPOINT_PREFIX = "colorwars_net"

# ─── Device ────────────────────────────────────────────────────────
# Auto-detected at runtime:
#   1. MPS (Apple Silicon)  — if available
#   2. CUDA (NVIDIA GPU)    — if available
#   3. CPU                  — fallback
