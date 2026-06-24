"""
model.py — AlphaZero-style dual-head CNN for Color Wars.

This is the neural network that learns to play Color Wars by studying
MCTS self-play data. It has two outputs:

  ┌──────────────────────────────────────────────────────────────┐
  │                    ColorWarsNet                              │
  │                                                              │
  │  Input: 5 planes × 6×6                                      │
  │     ↓                                                        │
  │  ConvBlock (64 filters, 3×3, BatchNorm, ReLU)               │
  │     ↓                                                        │
  │  ResBlock × 4 (64 filters, skip connections)                │
  │     ↓                                                        │
  │  ┌─────────────────┐     ┌─────────────────┐               │
  │  │  Policy Head     │     │   Value Head     │               │
  │  │  Conv 1×1 → BN   │     │  Conv 1×1 → BN   │               │
  │  │  → ReLU → FC(36) │     │  → ReLU → FC(64) │               │
  │  │  → log_softmax   │     │  → ReLU → FC(1)  │               │
  │  │                   │     │  → tanh           │               │
  │  └─────────────────┘     └─────────────────┘               │
  │                                                              │
  │  Policy: log-probabilities over 36 board cells              │
  │  Value:  scalar in [-1, +1] predicting game outcome         │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

Why dual-head?
──────────────
  • Policy head: "What move should I play?" — trained to match MCTS
    visit-count distributions. In Phase 5, this provides the prior
    probabilities for MCTS node selection.
  • Value head: "Who's winning?" — trained to predict game outcome
    from the current position. In Phase 5, this replaces random
    rollouts, making MCTS much more efficient.

Design Decisions:
─────────────────
  • log_softmax (not softmax) for numerical stability with NLLLoss
  • BatchNorm after every conv for training stability
  • Skip connections (ResBlocks) to enable gradient flow
  • MPS/CUDA auto-detection for hardware acceleration
  • Predict method handles single-state inference (no batch dim needed)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.neural_net import config


def get_device():
    """
    Auto-detect the best available compute device.

    Priority:
      1. CUDA (NVIDIA GPU)
      2. MPS (Apple Silicon GPU)
      3. CPU (fallback)

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ConvBlock(nn.Module):
    """
    Convolution → BatchNorm → ReLU.

    This is the basic building block used throughout the network.
    BatchNorm normalizes activations for stable training, and ReLU
    introduces the non-linearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """
    Residual block with skip connection.

    Architecture:
        x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU

    The skip connection (x added back) allows gradients to flow
    directly through the network, enabling deeper architectures
    without vanishing gradients. This is the same design used in
    AlphaZero and ResNet.
    """

    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        out = F.relu(out)
        return out


class ColorWarsNet(nn.Module):
    """
    AlphaZero-style neural network for Color Wars.

    Input:  Tensor of shape [batch, 5, 6, 6]
            (5 planes: P1 own, P1 dots, P2 own, P2 dots, current player)

    Output: Tuple (log_policy, value)
            - log_policy: [batch, 36] — log-probabilities over moves
            - value:      [batch, 1]  — game outcome prediction in [-1, +1]

    Usage:
        model = ColorWarsNet()
        board_tensor = torch.randn(1, 5, 6, 6)  # Single board
        log_policy, value = model(board_tensor)

        # For single-state inference (no batch dim):
        policy, value = model.predict(board_planes)
    """

    def __init__(self,
                 grid_size=config.GRID_SIZE,
                 num_input_planes=config.NUM_INPUT_PLANES,
                 num_res_blocks=config.NUM_RES_BLOCKS,
                 num_filters=config.NUM_FILTERS):
        """
        Args:
            grid_size:        Board size (6 for 6×6).
            num_input_planes: Number of input feature planes (5).
            num_res_blocks:   Number of residual blocks (4).
            num_filters:      Number of convolutional filters (64).
        """
        super().__init__()
        self.grid_size = grid_size
        self.policy_size = grid_size * grid_size

        # ── Input Convolution ──
        self.input_conv = ConvBlock(
            num_input_planes, num_filters,
            kernel_size=config.KERNEL_SIZE,
            padding=config.PADDING,
        )

        # ── Residual Tower ──
        self.res_blocks = nn.ModuleList([
            ResBlock(
                num_filters,
                kernel_size=config.KERNEL_SIZE,
                padding=config.PADDING,
            )
            for _ in range(num_res_blocks)
        ])

        # ── Policy Head ──
        # Conv 1×1 → BN → ReLU → Flatten → FC → log_softmax
        self.policy_conv = nn.Conv2d(
            num_filters, config.POLICY_FILTERS,
            kernel_size=1, bias=False,
        )
        self.policy_bn = nn.BatchNorm2d(config.POLICY_FILTERS)
        self.policy_fc = nn.Linear(
            config.POLICY_FILTERS * grid_size * grid_size,
            self.policy_size,
        )

        # ── Value Head ──
        # Conv 1×1 → BN → ReLU → Flatten → FC(64) → ReLU → FC(1) → tanh
        self.value_conv = nn.Conv2d(
            num_filters, config.VALUE_FILTERS,
            kernel_size=1, bias=False,
        )
        self.value_bn = nn.BatchNorm2d(config.VALUE_FILTERS)
        self.value_fc1 = nn.Linear(
            config.VALUE_FILTERS * grid_size * grid_size,
            config.VALUE_HIDDEN,
        )
        self.value_fc2 = nn.Linear(config.VALUE_HIDDEN, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch, 5, grid_size, grid_size].

        Returns:
            Tuple (log_policy, value):
              - log_policy: [batch, policy_size] — log-probabilities
              - value:      [batch, 1] — scalar in [-1, +1]
        """
        # ── Shared Trunk ──
        out = self.input_conv(x)
        for block in self.res_blocks:
            out = block(out)

        # ── Policy Head ──
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)  # Flatten
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # ── Value Head ──
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, board_planes, device=None):
        """
        Single-state inference — convenience method for agents.

        Takes a board encoding (list of 5 planes) and returns
        the policy probabilities and value estimate.

        Args:
            board_planes: List of 5 planes (each grid_size × grid_size).
                          Can be nested Python lists or numpy arrays.
            device:       Torch device. If None, uses the model's device.

        Returns:
            Tuple (policy_dict, value):
              - policy_dict: Dict mapping (row, col) → probability
              - value:       Float in [-1, +1]
        """
        self.eval()  # Inference mode

        if device is None:
            device = next(self.parameters()).device

        # Convert to tensor: [1, 5, grid_size, grid_size]
        import numpy as np
        board_np = np.array(board_planes, dtype=np.float32)
        board_tensor = torch.from_numpy(board_np).unsqueeze(0).to(device)

        with torch.no_grad():
            log_policy, value = self(board_tensor)

        # Convert log_policy to probabilities
        policy_probs = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value_scalar = value.item()

        # Build policy dict: (row, col) → probability
        policy_dict = {}
        for idx in range(self.policy_size):
            prob = float(policy_probs[idx])
            if prob > 1e-6:  # Skip negligible probabilities
                r = idx // self.grid_size
                c = idx % self.grid_size
                policy_dict[(r, c)] = prob

        return policy_dict, value_scalar

    def count_parameters(self):
        """
        Count total trainable parameters.

        Returns:
            Integer count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, filepath, epoch=None, optimizer_state=None,
                        loss=None, metadata=None):
        """
        Save model checkpoint to disk.

        Args:
            filepath:         Path to save the checkpoint file.
            epoch:            Current training epoch (optional).
            optimizer_state:  Optimizer state dict (optional).
            loss:             Current loss value (optional).
            metadata:         Additional metadata dict (optional).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'grid_size': self.grid_size,
            'config': {
                'num_res_blocks': len(self.res_blocks),
                'num_filters': self.input_conv.conv.out_channels,
                'num_input_planes': self.input_conv.conv.in_channels,
            },
        }

        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if loss is not None:
            checkpoint['loss'] = loss
        if metadata is not None:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(cls, filepath, device=None):
        """
        Load a model from a checkpoint file.

        Args:
            filepath: Path to the checkpoint file.
            device:   Device to load the model onto. If None, auto-detects.

        Returns:
            Tuple (model, checkpoint_data):
              - model: Loaded ColorWarsNet instance
              - checkpoint_data: Full checkpoint dict (with optimizer state, etc.)
        """
        if device is None:
            device = get_device()

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        # Reconstruct the model with the saved config
        model_config = checkpoint['config']
        model = cls(
            grid_size=checkpoint['grid_size'],
            num_input_planes=model_config['num_input_planes'],
            num_res_blocks=model_config['num_res_blocks'],
            num_filters=model_config['num_filters'],
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint

    def summary(self):
        """Print a compact model summary."""
        params = self.count_parameters()
        print(f"\n{'─' * 55}")
        print(f"  ColorWarsNet Summary")
        print(f"{'─' * 55}")
        print(f"  Input:          {config.NUM_INPUT_PLANES} planes × "
              f"{self.grid_size}×{self.grid_size}")
        print(f"  ResBlocks:      {len(self.res_blocks)}")
        print(f"  Filters:        {self.input_conv.conv.out_channels}")
        print(f"  Policy output:  {self.policy_size} moves")
        print(f"  Value output:   1 (tanh → [-1, +1])")
        print(f"  Parameters:     {params:,}")
        print(f"  Device:         {next(self.parameters()).device}")
        print(f"{'─' * 55}\n")
