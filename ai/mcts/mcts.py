"""
mcts.py — Monte Carlo Tree Search algorithm for Color Wars.

This implements the four phases of MCTS:

  ┌──────────────────────────────────────────────────────────┐
  │                    MCTS ALGORITHM                        │
  │                                                          │
  │  For each simulation:                                    │
  │                                                          │
  │  1. SELECTION                                            │
  │     Start at root. While node is fully expanded and      │
  │     not terminal, select child with highest UCB score.   │
  │                                                          │
  │  2. EXPANSION                                            │
  │     If node is not terminal, expand by adding one        │
  │     child for an untried action.                         │
  │                                                          │
  │  3. SIMULATION (Rollout)                                 │
  │     Play a random game from the new node to completion.  │
  │     In Phase 5, this is replaced by the neural network.  │
  │                                                          │
  │  4. BACKPROPAGATION                                      │
  │     Walk back up the tree, updating visit counts and     │
  │     values at each ancestor node.                        │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

After all simulations, pick the most-visited root child as the best move.

Performance Notes:
──────────────────
  • Each simulation is O(D × n²) where D = rollout depth, n = grid_size
  • Total search cost: O(S × D × n²) where S = number of simulations
  • Typical: 100-800 simulations per move for a 6×6 board
  • The rollout is the bottleneck — neural network evaluation replaces
    it in Phase 5, making search much more efficient.
"""

import random
from ai.mcts.mcts_node import MCTSNode
from ai.game_engine.constants import PLAYER_1, PLAYER_2


class MCTS:
    """
    Monte Carlo Tree Search engine.

    Usage:
        from ai.game_engine.game_state import GameState

        state = GameState()
        mcts = MCTS(n_simulations=200, c_puct=1.41)

        # Search and get the best move
        best_action, root = mcts.search(state)
        state.apply_move(*best_action)

        # Get move probabilities (for training the neural network)
        probs = mcts.get_action_probabilities(root, temperature=1.0)
    """

    def __init__(self, n_simulations=200, c_puct=1.41, rollout_depth=500,
                 seed=None):
        """
        Args:
            n_simulations: Number of MCTS simulations per search.
                           More = stronger but slower.
                           Typical: 100-800 for Color Wars.
            c_puct:        Exploration constant for UCB1.
                           Higher = more exploration.
                           1.41 ≈ √2 (classic UCB1).
            rollout_depth: Max moves in a random rollout before giving up.
                           Prevents infinite games.
            seed:          Random seed for reproducible rollouts.
        """
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    # ─── Main Search ─────────────────────────────────────────────────

    def search(self, state):
        """
        Run MCTS from the given state and return the best action.

        This is the main entry point. It:
          1. Creates a root node from the current state
          2. Runs N simulations (select → expand → rollout → backprop)
          3. Returns the most-visited child's action

        Args:
            state: Current GameState (not modified).

        Returns:
            Tuple (best_action, root_node):
              - best_action: (row, col) of the best move
              - root_node:   The root MCTSNode (for inspection/training data)

        Time: O(n_simulations × rollout_depth × grid_size²)
        """
        # Create root node from the current state
        root = MCTSNode(state=state.clone_state())

        # Run simulations
        for _ in range(self.n_simulations):
            self._run_simulation(root)

        # Pick the best move (most visited child)
        best_action, _ = root.best_action()
        return best_action, root

    # ─── Single Simulation ───────────────────────────────────────────

    def _run_simulation(self, root):
        """
        Execute one complete MCTS simulation:
        SELECT → EXPAND → ROLLOUT → BACKPROPAGATE.
        """
        node = root

        # ── 1. SELECTION ──
        # Walk down the tree, picking the best child at each level,
        # until we reach a node that hasn't been fully expanded
        # or is a terminal state.
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child(self.c_puct)

        # ── 2. EXPANSION ──
        # If the node is not terminal, expand it by adding one child.
        if not node.is_terminal:
            node = node.expand()

        # ── 3. SIMULATION (Rollout) ──
        # Play a random game from this node to completion.
        value = self._rollout(node.state, root.state.current_player)

        # ── 4. BACKPROPAGATION ──
        # Walk back up the tree, updating visit counts and values.
        node.backpropagate(value, root.state.current_player)

    # ─── Rollout (Random Playout) ────────────────────────────────────

    def _rollout(self, state, perspective_player):
        """
        Play a random game from the given state to completion.

        This is the SIMULATION phase — we play random moves until
        the game ends, then return the result.

        In Phase 5, this is replaced by neural network evaluation:
          value = neural_net.predict(state)
        For now, we use random rollouts.

        Args:
            state:              Starting state (will be cloned).
            perspective_player: Whose perspective to evaluate from.

        Returns:
            +1.0 if perspective_player wins
            -1.0 if perspective_player loses
             0.0 if game hits rollout_depth limit (rare, treated as draw)
        """
        # Clone so we don't modify the tree's state
        rollout_state = state.clone_state()

        for _ in range(self.rollout_depth):
            if rollout_state.is_game_over():
                break

            # Random move selection (fast rollout policy)
            moves = rollout_state.get_legal_moves()
            action = self.rng.choice(moves)
            rollout_state.apply_move(*action)

        # Evaluate the terminal state
        if rollout_state.is_game_over():
            winner = rollout_state.get_winner()
            if winner == perspective_player:
                return 1.0
            else:
                return -1.0

        # Hit depth limit — treat as draw
        return 0.0

    # ─── Action Probabilities (for NN training) ──────────────────────

    def get_action_probabilities(self, root, temperature=1.0):
        """
        Get move probabilities from visit counts at the root.

        These probabilities are used as training targets for the
        neural network's policy head (Phase 5).

        With temperature:
          • τ = 1.0: Proportional to visit counts (exploration)
          • τ → 0:   Picks the most visited move (exploitation)
          • τ > 1:   More uniform (more exploration)

        Formula:
          π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)

        Args:
            root:        The root MCTSNode after search.
            temperature: Controls exploration vs exploitation.
                         1.0 = proportional, 0.01 ≈ greedy.

        Returns:
            Dictionary mapping (row, col) → probability.
        """
        if not root.children:
            return {}

        if temperature < 0.01:
            # Near-zero temperature: pick the most visited child
            best_child = max(root.children, key=lambda c: c.visit_count)
            probs = {}
            for child in root.children:
                probs[child.action] = 1.0 if child is best_child else 0.0
            return probs

        # Apply temperature to visit counts
        visits = []
        for child in root.children:
            visits.append(child.visit_count ** (1.0 / temperature))

        total = sum(visits)
        if total == 0:
            # Uniform if no visits
            uniform = 1.0 / len(root.children)
            return {child.action: uniform for child in root.children}

        probs = {}
        for child, v in zip(root.children, visits):
            probs[child.action] = v / total

        return probs

    # ─── Diagnostics ─────────────────────────────────────────────────

    def search_diagnostics(self, root):
        """
        Print detailed diagnostics about the search tree.

        Shows the top moves, their visit counts, values, and UCB scores.
        Useful for understanding what MCTS is "thinking".

        Args:
            root: The root MCTSNode after search.
        """
        stats = root.tree_stats()
        print(f"\n{'─' * 55}")
        print(f"  MCTS SEARCH DIAGNOSTICS")
        print(f"{'─' * 55}")
        print(f"  Simulations:  {self.n_simulations}")
        print(f"  Tree nodes:   {stats['total_nodes']}")
        print(f"  Tree depth:   {stats['max_depth']}")
        print(f"  Root visits:  {stats['root_visits']}")
        print(f"  c_puct:       {self.c_puct}")
        print(f"{'─' * 55}")

        # Sort children by visit count
        sorted_children = sorted(
            root.children,
            key=lambda c: c.visit_count,
            reverse=True,
        )

        print(f"  {'Move':<10} {'Visits':>8} {'Value':>8} {'UCB':>8} {'Prior':>8}")
        print(f"  {'─' * 46}")

        for child in sorted_children[:10]:  # Top 10 moves
            ucb = child.ucb_score(self.c_puct)
            ucb_str = f"{ucb:.3f}" if ucb != float('inf') else "∞"
            print(f"  ({child.action[0]},{child.action[1]})"
                  f"{'':>5} {child.visit_count:>8} "
                  f"{child.q_value:>8.3f} {ucb_str:>8} "
                  f"{child.prior:>8.3f}")

        print(f"{'─' * 55}\n")
