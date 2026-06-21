"""
base_agent.py — Abstract base class for all Color Wars AI agents.

All agents (RandomAgent, MCTSAgent, NeuralNetAgent) must inherit from
this class and implement the `select_move()` method.

Design Pattern: Strategy Pattern
─────────────────────────────────
By defining a common interface, we can swap agents freely:
  • RandomAgent for baseline testing
  • MCTSAgent for tree-search-based play (Phase 3)
  • NeuralNetAgent for AlphaZero-style play (Phase 5)

The SelfPlayManager doesn't care which agent it's using — it just
calls `agent.select_move(state)` and gets back a valid move.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for Color Wars agents.

    Every agent must implement:
      • select_move(state) → (row, col)

    Usage:
        class MyAgent(BaseAgent):
            def select_move(self, state):
                moves = state.get_legal_moves()
                return moves[0]  # Always pick the first move

        agent = MyAgent(name="FirstMoveAgent")
        move = agent.select_move(game_state)
    """

    def __init__(self, name="BaseAgent"):
        """
        Args:
            name: Human-readable name for this agent (used in logs/stats).
        """
        self.name = name

    @abstractmethod
    def select_move(self, state):
        """
        Choose a move given the current game state.

        Args:
            state: A GameState object representing the current board.
                   The agent should NOT modify this state — use clone()
                   if simulation is needed.

        Returns:
            A tuple (row, col) representing the chosen move.
            Must be a valid move from state.get_legal_moves().

        Raises:
            ValueError: If no legal moves are available.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
