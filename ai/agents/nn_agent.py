"""
nn_agent.py — An agent that uses the trained neural network to play.

This agent wraps the ColorWarsNet model into the BaseAgent interface,
enabling it to play games via SelfPlayManager just like RandomAgent
or MCTSAgent.

Operating Modes:
────────────────
  1. Direct play: Uses the policy head output to select moves.
     Fast inference, no tree search. Good for quick games.

  2. Training mode: select_move_with_policy() returns both the
     chosen action and the full policy distribution. Compatible
     with SelfPlayManager's training data collection.

Move Selection:
───────────────
  • Temperature τ=0 (greedy): Pick the move with highest probability.
  • Temperature τ>0: Sample from the policy distribution.
  • Only legal moves are considered — illegal move probabilities are
    masked to zero and the distribution is renormalized.

Phase 5 Preview:
────────────────
  In Phase 5, this agent will be combined with MCTS: the neural network
  provides prior probabilities and value estimates, while MCTS provides
  the search. For now, this is a standalone NN agent.
"""

import random
import numpy as np
import torch

from ai.agents.base_agent import BaseAgent
from ai.neural_net.model import ColorWarsNet, get_device
from ai.self_play.board_encoder import encode_board, encode_board_from_perspective
from ai.game_engine.constants import PLAYER_1, PLAYER_2


class NNAgent(BaseAgent):
    """
    Agent that uses a trained neural network to select moves.

    Usage:
        # Load a trained model
        agent = NNAgent.from_checkpoint("ai/neural_net/checkpoints/model.pt")

        # Or wrap an existing model
        model = ColorWarsNet()
        agent = NNAgent(model=model)

        # Play a move
        state = GameState()
        move = agent.select_move(state)
    """

    def __init__(self, model, name="NNAgent", temperature=0.01,
                 temperature_threshold=None, use_perspective=True,
                 device=None, seed=None):
        """
        Args:
            model:                 Trained ColorWarsNet instance.
            name:                  Agent name for logging.
            temperature:           Move selection temperature.
                                   0.01 = near-greedy (best for competitive play).
                                   1.0 = proportional (for generating training data).
            temperature_threshold: Move number at which temperature drops to
                                   near-zero. None = fixed temperature.
            use_perspective:       If True, encode board from current player's
                                   perspective (canonical view).
            device:                Torch device. If None, uses model's device.
            seed:                  Random seed for reproducible play.
        """
        super().__init__(name=name)
        self.model = model
        self.model.eval()
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.use_perspective = use_perspective
        self.device = device or next(model.parameters()).device
        self.grid_size = model.grid_size
        self._rng = random.Random(seed)

        # Cache last prediction for diagnostics
        self.last_policy = None
        self.last_value = None

    @classmethod
    def from_checkpoint(cls, filepath, **kwargs):
        """
        Create an NNAgent from a saved checkpoint.

        Args:
            filepath: Path to the checkpoint file.
            **kwargs: Additional arguments passed to NNAgent.__init__.

        Returns:
            NNAgent instance with the loaded model.
        """
        model, _ = ColorWarsNet.load_checkpoint(filepath)
        return cls(model=model, **kwargs)

    def _get_temperature(self, move_number=None):
        """
        Get effective temperature for the given move number.

        Same annealing strategy as MCTSAgent.
        """
        if self.temperature_threshold is None or move_number is None:
            return self.temperature
        if move_number <= self.temperature_threshold:
            return self.temperature
        return 0.01

    def _encode_state(self, state):
        """
        Encode a GameState into the neural network's input format.

        Args:
            state: GameState object.

        Returns:
            List of 5 planes (grid_size × grid_size each).
        """
        if self.use_perspective:
            return encode_board_from_perspective(
                state.board, state.current_player, self.grid_size,
            )
        else:
            return encode_board(
                state.board, state.current_player, self.grid_size,
            )

    def _get_policy(self, state):
        """
        Get the neural network's policy for the current state.

        Masks illegal moves and renormalizes the distribution.

        Args:
            state: GameState object.

        Returns:
            Dict mapping (row, col) → probability (only legal moves).
        """
        board_planes = self._encode_state(state)

        # Get raw predictions
        raw_policy, value = self.model.predict(board_planes, device=self.device)
        self.last_value = value

        # Get legal moves
        legal_moves = set(state.get_legal_moves())

        # Mask illegal moves
        masked_policy = {}
        total = 0.0
        for action, prob in raw_policy.items():
            if action in legal_moves:
                masked_policy[action] = prob
                total += prob

        # Renormalize
        if total > 0:
            for action in masked_policy:
                masked_policy[action] /= total
        else:
            # Fallback: uniform over legal moves
            uniform = 1.0 / len(legal_moves)
            masked_policy = {move: uniform for move in legal_moves}

        self.last_policy = masked_policy
        return masked_policy

    def select_move(self, state):
        """
        Use the neural network to select the best move.

        Args:
            state: Current GameState (not modified).

        Returns:
            Tuple (row, col) — the selected move.
        """
        action, _ = self.select_move_with_policy(state)
        return action

    def select_move_with_policy(self, state, move_number=None):
        """
        Use the neural network to select a move and return the policy.

        Compatible with SelfPlayManager's training data collection.

        Args:
            state:       Current GameState (not modified).
            move_number: Current move number for temperature annealing.

        Returns:
            Tuple (action, policy):
              - action: (row, col) of the selected move
              - policy: dict mapping (row, col) → probability
        """
        moves = state.get_legal_moves()
        if not moves:
            raise ValueError(f"{self.name}: No legal moves available.")

        if len(moves) == 1:
            policy = {moves[0]: 1.0}
            self.last_policy = policy
            self.last_value = None
            return moves[0], policy

        # Get the neural network's policy
        policy = self._get_policy(state)

        # Apply temperature
        effective_temp = self._get_temperature(move_number)

        if effective_temp < 0.02:
            # Greedy: pick the highest probability move
            best_action = max(policy, key=policy.get)
            return best_action, policy
        else:
            # Sample from the distribution with temperature
            actions = list(policy.keys())
            probs = np.array([policy[a] for a in actions], dtype=np.float64)

            # Apply temperature
            if effective_temp != 1.0:
                probs = probs ** (1.0 / effective_temp)

            # Renormalize
            total = probs.sum()
            if total > 0:
                probs = probs / total
            else:
                probs = np.ones(len(actions)) / len(actions)

            # Sample
            selected_idx = self._rng.choices(
                range(len(actions)),
                weights=probs.tolist(),
                k=1,
            )[0]
            return actions[selected_idx], policy

    def get_value_estimate(self, state):
        """
        Get the neural network's value estimate for a state.

        Args:
            state: GameState object.

        Returns:
            Float in [-1, +1] — estimated game outcome.
        """
        board_planes = self._encode_state(state)
        _, value = self.model.predict(board_planes, device=self.device)
        return value
