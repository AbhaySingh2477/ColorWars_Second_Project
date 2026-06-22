"""
mcts_node.py — Node in the Monte Carlo Search Tree.

Each node represents a game state reached by a specific action.
The tree grows during search as we explore promising moves.

MCTS Tree Structure:
────────────────────
    Root (current game state)
    ├── Child 1 (after move A)  ← visit_count=50, total_value=28
    │   ├── Grandchild 1a
    │   └── Grandchild 1b
    ├── Child 2 (after move B)  ← visit_count=30, total_value=20
    └── Child 3 (after move C)  ← visit_count=20, total_value=5

Each node stores:
  • state:        The game state at this node
  • action:       The move that led to this node (None for root)
  • parent:       The parent node (None for root)
  • children:     List of child nodes (expanded moves)
  • visit_count:  How many times this node has been visited (N)
  • total_value:  Sum of backpropagated values (W)
  • prior:        Prior probability from the neural network (P)
                  (For Phase 3 we use uniform prior; Phase 5 adds the NN)

Key Formulas:
─────────────
  • Q(s,a) = W(s,a) / N(s,a)       — average value (exploitation)
  • U(s,a) = c_puct × P(s,a) × √(N_parent) / (1 + N(s,a))  — exploration
  • Score  = Q(s,a) + U(s,a)         — UCB1 selection criterion

  Where:
    N = visit count
    W = total value
    P = prior probability
    c_puct = exploration constant (typically 1.0–2.0)
"""

import math
from ai.game_engine.constants import PLAYER_1, PLAYER_2


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.

    Attributes:
        state:          GameState at this node.
        action:         The (row, col) move that created this node (None for root).
        parent:         Parent MCTSNode (None for root).
        children:       List of child MCTSNodes.
        visit_count:    Number of times visited during search (N).
        total_value:    Cumulative value from simulations (W).
        prior:          Prior probability for this action (P).
        untried_actions: Legal moves not yet expanded into children.
        player_who_moved: The player who made the action to reach this state.
    """

    __slots__ = [
        'state', 'action', 'parent', 'children',
        'visit_count', 'total_value', 'prior',
        'untried_actions', 'player_who_moved',
    ]

    def __init__(self, state, action=None, parent=None, prior=0.0):
        """
        Args:
            state:  GameState at this node.
            action: The move that led here (None for root).
            parent: Parent MCTSNode (None for root).
            prior:  Prior probability from policy network (uniform in Phase 3).
        """
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.untried_actions = state.get_legal_moves() if not state.is_game_over() else []

        # Track who moved to reach this node
        # (the opponent of the current player, since turns alternate)
        if parent is not None:
            self.player_who_moved = PLAYER_2 if state.current_player == PLAYER_1 else PLAYER_1
        else:
            self.player_who_moved = None  # Root has no incoming move

    # ─── Properties ──────────────────────────────────────────────────

    @property
    def q_value(self):
        """
        Average value of this node: Q = W / N.

        Returns 0 if unvisited to avoid division by zero.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_fully_expanded(self):
        """True if all legal moves have been expanded into children."""
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self):
        """True if this is a game-over state (leaf of the actual game)."""
        return self.state.is_game_over()

    @property
    def is_root(self):
        """True if this is the root of the search tree."""
        return self.parent is None

    # ─── UCB1 Score ──────────────────────────────────────────────────

    def ucb_score(self, c_puct=1.41):
        """
        Compute the Upper Confidence Bound score for selection.

        Score = Q(s,a) + c_puct × P(s,a) × √(N_parent) / (1 + N(s,a))

        • High Q → exploit nodes that have performed well
        • High U → explore nodes that haven't been visited much
        • c_puct controls the exploration-exploitation tradeoff

        Args:
            c_puct: Exploration constant. Higher = more exploration.
                    - 1.41 ≈ √2 (classic UCB1)
                    - AlphaZero uses ~1.0–2.5

        Returns:
            Float score for this node. Higher is better for selection.
        """
        if self.visit_count == 0:
            # Unvisited nodes get maximum priority (always explore first)
            return float('inf')

        parent_visits = self.parent.visit_count if self.parent else 1

        # Exploitation: average reward
        exploitation = self.q_value

        # Exploration: bonus for under-visited nodes
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

        return exploitation + exploration

    # ─── Tree Operations ─────────────────────────────────────────────

    def expand(self):
        """
        Expand the tree by adding a child node for one untried action.

        Pops an action from untried_actions, creates the resulting game
        state, wraps it in a new MCTSNode, and adds it as a child.

        Returns:
            The newly created child MCTSNode.

        Raises:
            RuntimeError: If no untried actions remain.
        """
        if not self.untried_actions:
            raise RuntimeError("No untried actions to expand")

        action = self.untried_actions.pop()

        # Create the new state by applying the move
        child_state = self.state.clone_state()
        child_state.apply_move(*action)

        # Uniform prior for Phase 3 (neural network replaces this in Phase 5)
        n_legal = len(self.untried_actions) + len(self.children) + 1
        uniform_prior = 1.0 / n_legal if n_legal > 0 else 1.0

        child_node = MCTSNode(
            state=child_state,
            action=action,
            parent=self,
            prior=uniform_prior,
        )

        self.children.append(child_node)
        return child_node

    def best_child(self, c_puct=1.41):
        """
        Select the child with the highest UCB score.

        This is the SELECTION step of MCTS — we pick the most
        promising child to continue searching from.

        Args:
            c_puct: Exploration constant for UCB.

        Returns:
            The child MCTSNode with the highest UCB score.
        """
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def best_action(self):
        """
        Select the best action based on visit counts (not UCB).

        After search is complete, we pick the most-visited child.
        Visit count is a more robust indicator than average value
        because it represents how much "confidence" the search has
        in that move.

        Returns:
            Tuple (action, child_node) — the best move and its node.
        """
        if not self.children:
            raise RuntimeError("No children to select from")

        best = max(self.children, key=lambda c: c.visit_count)
        return best.action, best

    def backpropagate(self, value, perspective_player):
        """
        Backpropagate a simulation result up the tree.

        The value is from the perspective of `perspective_player`.
        Each node flips the value depending on whether the node's
        player matches the perspective player.

        Args:
            value: The simulation outcome (+1 win, -1 loss, 0 draw)
                   from perspective_player's viewpoint.
            perspective_player: The player whose perspective the value is from.
        """
        node = self
        while node is not None:
            node.visit_count += 1

            # If the player who moved to reach this node matches
            # the perspective player, the value is positive for them.
            # Otherwise, it's negative (their opponent won).
            if node.player_who_moved == perspective_player:
                node.total_value += value
            elif node.player_who_moved is not None:
                node.total_value -= value

            node = node.parent

    # ─── Diagnostics ─────────────────────────────────────────────────

    def tree_stats(self):
        """
        Get statistics about the search tree rooted at this node.

        Returns:
            Dictionary with tree statistics.
        """
        total_nodes = 0
        max_depth = 0

        def _count(node, depth):
            nonlocal total_nodes, max_depth
            total_nodes += 1
            max_depth = max(max_depth, depth)
            for child in node.children:
                _count(child, depth + 1)

        _count(self, 0)

        return {
            'total_nodes': total_nodes,
            'max_depth': max_depth,
            'root_visits': self.visit_count,
            'children_count': len(self.children),
        }

    def __repr__(self):
        return (
            f"MCTSNode(action={self.action}, "
            f"visits={self.visit_count}, "
            f"value={self.q_value:.3f}, "
            f"children={len(self.children)})"
        )
