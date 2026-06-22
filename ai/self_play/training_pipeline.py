"""
training_pipeline.py — End-to-end training data generation for Phase 3.

This module orchestrates the complete MCTS self-learning pipeline:
  1. Generate training data via MCTS self-play
  2. Load and aggregate data from saved game records
  3. Extract and augment training examples
  4. Prepare data for neural network training (Phase 4)

Pipeline Architecture:
──────────────────────
  TrainingPipeline
    ├── generate(n_games, n_sims, ...)      → Run MCTS self-play
    ├── load_games(data_dir)                → Load saved GameRecords
    ├── extract_examples(records, augment)  → (board, policy, value) triples
    ├── sample_batch(examples, batch_size)  → Random training batch
    └── get_data_stats(examples)            → Statistics about the data

Data Flow:
──────────
  MCTSAgent × 2
      ↓
  SelfPlayManager.run_training_self_play()
      ↓
  List[GameRecord]  (with policy data)
      ↓
  extract_examples()  →  [(board, policy, outcome), ...]
      ↓
  augment  →  8× more examples via symmetry
      ↓
  sample_batch()  →  ready for neural network

Training Example Format:
────────────────────────
  Each example is a tuple (board_state, policy, value):
    • board_state: 2D list [owner, dots] — raw game board
    • policy:      Dict {(row, col): prob} — MCTS search distribution
    • value:       +1.0 (win) or -1.0 (loss) from this player's perspective

  For neural network consumption (Phase 4), use board_encoder:
    • encode_board()  → 5-plane tensor input
    • encode_policy() → flat probability vector target
    • value          → scalar target
"""

import os
import time
import random

from ai.agents.mcts_agent import MCTSAgent
from ai.self_play.self_play_manager import SelfPlayManager
from ai.self_play.game_record import GameRecord
from ai.self_play.board_encoder import (
    encode_board,
    encode_board_from_perspective,
    encode_policy,
    augment_training_example,
)
from ai.game_engine.constants import GRID_SIZE, PLAYER_1, PLAYER_2


class TrainingPipeline:
    """
    End-to-end training data pipeline for MCTS self-learning.

    This class manages the full cycle:
      1. Generate MCTS self-play games
      2. Extract training examples
      3. Augment via symmetry
      4. Prepare batches for training

    Usage:
        pipeline = TrainingPipeline(grid_size=6)

        # Generate fresh training data
        records = pipeline.generate(
            n_games=50,
            n_simulations=200,
            save_dir="data/mcts_self_play",
        )

        # Extract and augment training examples
        examples = pipeline.extract_examples(records, augment=True)
        print(f"Generated {len(examples)} training examples")

        # Sample a training batch
        batch = pipeline.sample_batch(examples, batch_size=32)

        # Or load previously saved games
        records = pipeline.load_games("data/mcts_self_play")
        examples = pipeline.extract_examples(records, augment=True)
    """

    def __init__(self, grid_size=GRID_SIZE):
        """
        Args:
            grid_size: Size of the game grid (default: 6).
        """
        self.grid_size = grid_size

    # ─── Generate Training Data ──────────────────────────────────────

    def generate(self, n_games, n_simulations=200, c_puct=1.41,
                 temperature=1.0, temperature_threshold=8,
                 save_dir=None, seed=None, verbose=False,
                 swap_colors=True):
        """
        Generate training data via MCTS self-play.

        Creates two MCTSAgent instances and runs them against each other
        with full policy data collection.

        Args:
            n_games:               Number of self-play games.
            n_simulations:         MCTS simulations per move.
            c_puct:                Exploration constant for UCB.
            temperature:           Initial temperature for move selection.
            temperature_threshold: Move number at which temp drops to greedy.
            save_dir:              Directory to save game records.
            seed:                  Random seed for reproducibility.
            verbose:               Print detailed output.
            swap_colors:           Alternate P1/P2 between games.

        Returns:
            List of GameRecord objects with MCTS policy data.
        """
        # Create two MCTS agents for self-play
        agent1 = MCTSAgent(
            name="MCTS-A",
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature=temperature,
            temperature_threshold=temperature_threshold,
            seed=seed,
        )
        agent2 = MCTSAgent(
            name="MCTS-B",
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature=temperature,
            temperature_threshold=temperature_threshold,
            seed=seed + 7919 if seed is not None else None,
        )

        # Run self-play with policy collection
        manager = SelfPlayManager(grid_size=self.grid_size)
        records = manager.run_training_self_play(
            agent1=agent1,
            agent2=agent2,
            n_games=n_games,
            save_dir=save_dir,
            verbose=verbose,
            swap_colors=swap_colors,
        )

        # Print summary
        stats = manager.get_statistics(records)
        manager.print_statistics(stats)

        return records

    # ─── Load Saved Games ────────────────────────────────────────────

    @staticmethod
    def load_games(data_dir, max_games=None):
        """
        Load saved GameRecord files from a directory.

        Loads all .json files in the directory, sorted by name.

        Args:
            data_dir:  Directory containing game JSON files.
            max_games: Maximum number of games to load (None = all).

        Returns:
            List of GameRecord objects.

        Raises:
            FileNotFoundError: If the directory doesn't exist.
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}"
            )

        files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith('.json')
        )

        if max_games is not None:
            files = files[:max_games]

        records = []
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            records.append(GameRecord.load(filepath))

        print(f"  Loaded {len(records)} games from {data_dir}")

        policy_count = sum(1 for r in records if r.has_policy)
        if policy_count > 0:
            print(f"  Games with policy data: {policy_count}/{len(records)}")

        return records

    # ─── Extract Training Examples ───────────────────────────────────

    def extract_examples(self, records, augment=False,
                         policy_only=False):
        """
        Extract training examples from game records.

        Iterates over all games and all moves, producing training
        examples of the form (board_state, policy, outcome).

        Args:
            records:     List of GameRecord objects.
            augment:     If True, apply 8-fold symmetry augmentation.
                         Multiplies examples by 8×.
            policy_only: If True, skip moves without MCTS policy data.
                         Use this to filter out RandomAgent moves.

        Returns:
            List of (board_state, action_probs, outcome) tuples:
              - board_state:   2D list [owner, dots]
              - action_probs:  Dict {(r,c): prob} or None
              - outcome:       +1.0 (win) or -1.0 (loss)
        """
        all_examples = []

        for record in records:
            examples = record.get_training_examples()

            for board_state, action_probs, outcome in examples:
                # Skip non-policy examples if requested
                if policy_only and action_probs is None:
                    continue

                if augment and action_probs is not None:
                    # Generate 8 augmented versions
                    augmented = augment_training_example(
                        board_state, action_probs, outcome,
                        self.grid_size,
                    )
                    all_examples.extend(augmented)
                else:
                    all_examples.append((board_state, action_probs, outcome))

        return all_examples

    # ─── Encode Examples for Neural Network ──────────────────────────

    def encode_examples(self, examples, perspective=False):
        """
        Encode raw training examples into neural-network-ready format.

        Converts board states to 5-plane encoded representation and
        policies to flat probability vectors.

        Args:
            examples:    List of (board_state, action_probs, outcome).
            perspective: If True, encode from the current player's
                         perspective (canonical view). If False, use
                         fixed P1/P2 planes.

        Returns:
            List of (encoded_board, flat_policy, value) tuples:
              - encoded_board: List of 5 planes (grid_size × grid_size each)
              - flat_policy:   List of grid_size² probabilities
              - value:         +1.0 or -1.0

        Note: The current player is inferred from the board state by
        checking who has more moves (simplified — for full accuracy,
        the GameRecord should store the current_player, which we'll
        add in Phase 4 if needed).
        """
        encoded = []

        for board_state, action_probs, outcome in examples:
            # Infer current player from move count parity
            # (this is an approximation — works for our training pipeline)
            p1_cells = sum(
                1 for row in board_state for cell in row if cell[0] == PLAYER_1
            )
            p2_cells = sum(
                1 for row in board_state for cell in row if cell[0] == PLAYER_2
            )
            # If equal cells or P1 has more, it's likely P1's turn in
            # the early game. For training, this approximation is fine.
            current_player = PLAYER_1 if p1_cells <= p2_cells else PLAYER_2

            if perspective:
                board_planes = encode_board_from_perspective(
                    board_state, current_player, self.grid_size
                )
            else:
                board_planes = encode_board(
                    board_state, current_player, self.grid_size
                )

            flat_policy = encode_policy(action_probs, self.grid_size)
            encoded.append((board_planes, flat_policy, outcome))

        return encoded

    # ─── Sample Training Batch ───────────────────────────────────────

    @staticmethod
    def sample_batch(examples, batch_size, rng=None):
        """
        Sample a random batch of training examples.

        Args:
            examples:   List of training examples (any format).
            batch_size: Number of examples to sample.
            rng:        Optional random.Random instance for reproducibility.

        Returns:
            List of batch_size randomly sampled examples.
            If batch_size > len(examples), returns all examples shuffled.
        """
        if rng is None:
            rng = random.Random()

        if batch_size >= len(examples):
            shuffled = list(examples)
            rng.shuffle(shuffled)
            return shuffled

        return rng.sample(examples, batch_size)

    # ─── Data Statistics ─────────────────────────────────────────────

    @staticmethod
    def get_data_stats(examples):
        """
        Compute statistics about training examples.

        Args:
            examples: List of (board_state, action_probs, outcome) tuples.

        Returns:
            Dictionary with:
              - total_examples: Total number of examples
              - with_policy: Number with MCTS policy data
              - without_policy: Number without policy (RandomAgent)
              - positive_outcomes: Examples where player won (+1)
              - negative_outcomes: Examples where player lost (-1)
              - outcome_balance: Ratio of positive to negative outcomes
        """
        total = len(examples)
        with_policy = sum(1 for _, p, _ in examples if p is not None)
        positives = sum(1 for _, _, o in examples if o > 0)
        negatives = total - positives

        return {
            'total_examples': total,
            'with_policy': with_policy,
            'without_policy': total - with_policy,
            'positive_outcomes': positives,
            'negative_outcomes': negatives,
            'outcome_balance': (
                positives / negatives if negatives > 0
                else float('inf') if positives > 0
                else 0.0
            ),
        }

    @staticmethod
    def print_data_stats(stats):
        """
        Pretty-print training data statistics.

        Args:
            stats: Dictionary from get_data_stats().
        """
        print(f"\n{'─' * 55}")
        print(f"  TRAINING DATA STATISTICS")
        print(f"{'─' * 55}")
        print(f"  Total examples:      {stats['total_examples']}")
        print(f"  With MCTS policy:    {stats['with_policy']}")
        print(f"  Without policy:      {stats['without_policy']}")
        print(f"{'─' * 55}")
        print(f"  Positive (+1 wins):  {stats['positive_outcomes']}")
        print(f"  Negative (-1 loss):  {stats['negative_outcomes']}")
        print(f"  Outcome balance:     {stats['outcome_balance']:.2f}")
        print(f"{'═' * 55}\n")
