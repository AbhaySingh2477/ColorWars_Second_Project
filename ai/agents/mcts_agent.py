"""
mcts_agent.py — An agent that uses Monte Carlo Tree Search.

This agent wraps the MCTS algorithm into the BaseAgent interface so it
can be plugged into the SelfPlayManager just like RandomAgent.

Performance vs RandomAgent:
───────────────────────────
  With just 100 simulations per move, MCTS should already beat
  RandomAgent ~85-95% of the time. With 400+, it's nearly unbeatable
  by random play.

Key Parameters:
  • n_simulations: More = stronger. 100-200 is fast, 400-800 is strong.
  • c_puct: Exploration constant. 1.41 is standard.
  • temperature: Controls move selection randomness.
    - 1.0: Proportional to visit counts (good for training data)
    - 0.01: Near-greedy (good for competitive play)
  • temperature_threshold: Number of moves before temperature drops to
    near-zero. Enables exploration in opening, exploitation in midgame.
    AlphaZero uses 30 for Go (19×19); we default to 8 for Color Wars (6×6).

Phase 3 Additions:
──────────────────
  • select_move_with_policy(): Returns both the action AND the MCTS
    policy distribution (for training data collection).
  • Temperature annealing: Automatically switches from exploratory
    (τ=1.0) to greedy (τ→0) after temperature_threshold moves.
  • Policy caching: Stores the last policy for retrieval by
    SelfPlayManager without changing the select_move() API.
"""

import random
from ai.agents.base_agent import BaseAgent
from ai.mcts.mcts import MCTS


class MCTSAgent(BaseAgent):
    """
    Agent that uses Monte Carlo Tree Search to select moves.

    Can operate in two modes:
      1. Competitive play: select_move() — returns just the best action
      2. Training mode: select_move_with_policy() — returns action + policy

    Usage:
        # Competitive play
        agent = MCTSAgent(n_simulations=200)
        move = agent.select_move(state)

        # Training data collection
        agent = MCTSAgent(n_simulations=200, temperature=1.0,
                          temperature_threshold=8)
        move, policy = agent.select_move_with_policy(state, move_number=5)
    """

    def __init__(self, name="MCTSAgent", n_simulations=200, c_puct=1.41,
                 temperature=0.01, temperature_threshold=None, seed=None):
        """
        Args:
            name:                  Agent name for logging.
            n_simulations:         MCTS simulations per move.
            c_puct:                Exploration constant for UCB.
            temperature:           Move selection temperature.
                                   0.01 = near-greedy (best for competitive play).
                                   1.0 = proportional (for generating training data).
            temperature_threshold: Move number at which temperature drops to
                                   near-zero (0.01). None = fixed temperature.
                                   For training: set to ~8 for 6×6 board.
                                   Moves 1..threshold use the given temperature.
                                   Moves threshold+1.. use greedy (τ=0.01).
            seed:                  Random seed for reproducible play.
        """
        super().__init__(name=name)
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.seed = seed
        self.mcts = MCTS(
            n_simulations=n_simulations,
            c_puct=c_puct,
            seed=seed,
        )
        # Store the last search root for diagnostics
        self.last_root = None
        # Store the last policy for retrieval by SelfPlayManager
        self.last_policy = None
        # Random instance for temperature-based selection
        self._rng = random.Random(seed)

    def _get_temperature(self, move_number=None):
        """
        Get the effective temperature for a given move number.

        Implements AlphaZero-style temperature annealing:
          • Moves 1 to threshold: use self.temperature (exploration)
          • Moves > threshold: use 0.01 (near-greedy exploitation)

        Args:
            move_number: Current move number (1-indexed). None = use
                         the base temperature always.

        Returns:
            Float temperature value.
        """
        if self.temperature_threshold is None or move_number is None:
            return self.temperature

        if move_number <= self.temperature_threshold:
            return self.temperature
        else:
            return 0.01  # Near-greedy

    def select_move(self, state):
        """
        Use MCTS to select the best move.

        For backward compatibility, this method does NOT return policy data.
        Use select_move_with_policy() for training data collection.

        Args:
            state: Current GameState (not modified).

        Returns:
            Tuple (row, col) — the selected move.
        """
        action, _ = self.select_move_with_policy(state)
        return action

    def select_move_with_policy(self, state, move_number=None):
        """
        Use MCTS to select a move AND return the search policy.

        This is the primary method for training data collection.
        It returns both the chosen action and the full MCTS policy
        distribution (visit-count-based probabilities over all actions).

        Args:
            state:       Current GameState (not modified).
            move_number: Current move number (1-indexed) for temperature
                         annealing. None = use base temperature.

        Returns:
            Tuple (action, policy):
              - action: (row, col) of the selected move
              - policy: dict mapping (row, col) → probability
                        Sum of probabilities ≈ 1.0

        The policy is the MCTS visit-count distribution, which is
        the training target for the neural network's policy head.
        """
        moves = state.get_legal_moves()
        if not moves:
            raise ValueError(f"{self.name}: No legal moves available.")

        # If only one legal move, no need to search
        if len(moves) == 1:
            policy = {moves[0]: 1.0}
            self.last_policy = policy
            self.last_root = None
            return moves[0], policy

        # Run MCTS search
        best_action, root = self.mcts.search(state)
        self.last_root = root

        # Get the effective temperature for this move
        effective_temp = self._get_temperature(move_number)

        # Compute the policy (action probability distribution)
        policy = self.mcts.get_action_probabilities(root, temperature=1.0)
        self.last_policy = policy

        # Select the action based on the effective temperature
        if effective_temp > 0.02:
            # Exploration: sample proportionally from the policy
            # (but use the actual temperature for selection, not the
            # stored policy which is always computed at τ=1.0)
            selection_probs = self.mcts.get_action_probabilities(
                root, temperature=effective_temp
            )
            actions = list(selection_probs.keys())
            weights = list(selection_probs.values())
            selected = self._rng.choices(actions, weights=weights, k=1)[0]
            return selected, policy
        else:
            # Exploitation: pick the most visited child
            return best_action, policy

    def get_search_stats(self):
        """
        Get statistics from the last search.

        Returns:
            Dictionary with tree stats, or None if no search has been run.
        """
        if self.last_root is None:
            return None
        return self.last_root.tree_stats()

    def print_diagnostics(self):
        """Print detailed search diagnostics from the last move."""
        if self.last_root is not None:
            self.mcts.search_diagnostics(self.last_root)
