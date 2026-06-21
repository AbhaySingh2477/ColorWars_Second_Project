"""
random_agent.py — An agent that picks moves uniformly at random.

Purpose:
────────
  • Baseline agent for testing the self-play infrastructure.
  • Provides a "worst case" opponent — any trained AI should beat this easily.
  • Used to generate initial training data before the neural network exists.
  • Useful for benchmarking: "How much better is MCTS than random?"

Algorithm:
──────────
  1. Get all legal moves from the game state.
  2. Pick one uniformly at random.
  3. Return it.

Time Complexity:  O(n²) — dominated by get_legal_moves()
Space Complexity: O(n²) — the list of legal moves
"""

import random
from ai.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that selects moves uniformly at random.

    This is the simplest possible agent. It serves as:
      • A baseline to compare smarter agents against
      • A data generator for initial neural network training
      • A test opponent for the self-play system

    Usage:
        agent = RandomAgent()
        state = GameState()
        move = agent.select_move(state)
        state.apply_move(*move)
    """

    def __init__(self, name="RandomAgent", seed=None):
        """
        Args:
            name: Human-readable name for logging.
            seed: Optional random seed for reproducibility.
                  If None, uses Python's default (system entropy).
        """
        super().__init__(name=name)
        # Each agent gets its own Random instance so multiple agents
        # don't interfere with each other's random streams.
        self.rng = random.Random(seed)

    def select_move(self, state):
        """
        Pick a random legal move.

        Args:
            state: Current GameState (not modified).

        Returns:
            Tuple (row, col) — a randomly chosen legal move.

        Raises:
            ValueError: If no legal moves exist (game is over).
        """
        moves = state.get_legal_moves()

        if not moves:
            raise ValueError(
                f"{self.name}: No legal moves available. "
                f"Game state: {state}"
            )

        return self.rng.choice(moves)
