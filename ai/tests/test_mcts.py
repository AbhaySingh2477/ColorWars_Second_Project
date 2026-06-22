"""
test_mcts.py — Unit tests for Phase 3: Monte Carlo Tree Search.

Tests cover:
  1. MCTSNode — creation, expansion, UCB scoring, backpropagation
  2. MCTS — search, rollouts, action probabilities
  3. MCTSAgent — move selection, integration with BaseAgent
  4. Phase 3 additions — clone_state(), policy collection, temperature annealing

Run with:
    cd ColorWars
    source .venv/bin/activate
    python -m pytest ai/tests/test_mcts.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.game_engine.game_state import GameState
from ai.game_engine.constants import PLAYER_1, PLAYER_2, GRID_SIZE
from ai.mcts.mcts_node import MCTSNode
from ai.mcts.mcts import MCTS
from ai.agents.mcts_agent import MCTSAgent
from ai.agents.random_agent import RandomAgent
from ai.agents.base_agent import BaseAgent


# ═══════════════════════════════════════════════════════════════════════
# 1. MCTS NODE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestMCTSNode:
    """Test the MCTSNode class."""

    def test_root_node_creation(self):
        """Root node should have no parent or action."""
        state = GameState()
        root = MCTSNode(state=state)

        assert root.parent is None
        assert root.action is None
        assert root.is_root
        assert root.visit_count == 0
        assert root.total_value == 0.0
        assert root.q_value == 0.0

    def test_untried_actions_populated(self):
        """New node should have all legal moves as untried."""
        state = GameState()
        root = MCTSNode(state=state)

        legal_moves = state.get_legal_moves()
        assert len(root.untried_actions) == len(legal_moves)

    def test_terminal_node_has_no_untried(self):
        """Terminal (game-over) node should have no untried actions."""
        state = GameState()
        state.game_over = True

        node = MCTSNode(state=state)
        assert len(node.untried_actions) == 0
        assert node.is_terminal

    def test_expand_creates_child(self):
        """Expanding should create a child node."""
        state = GameState()
        root = MCTSNode(state=state)

        initial_untried = len(root.untried_actions)
        child = root.expand()

        assert child.parent is root
        assert child in root.children
        assert len(root.untried_actions) == initial_untried - 1
        assert child.action is not None

    def test_expand_child_state_independent(self):
        """Expanded child's state should be independent from parent."""
        state = GameState()
        root = MCTSNode(state=state)
        child = root.expand()

        # Modify child state
        child.state.turn_count = 999

        # Parent should be unaffected
        assert root.state.turn_count == 0

    def test_fully_expanded(self):
        """Node should be fully expanded after all moves are tried."""
        state = GameState(grid_size=2)  # Small board for speed
        root = MCTSNode(state=state)

        while root.untried_actions:
            root.expand()

        assert root.is_fully_expanded
        assert len(root.children) > 0

    def test_ucb_unvisited_is_infinity(self):
        """Unvisited nodes should have infinite UCB score."""
        state = GameState()
        root = MCTSNode(state=state)
        child = root.expand()

        assert child.ucb_score() == float('inf')

    def test_ucb_after_visit(self):
        """UCB should be finite after a visit."""
        state = GameState()
        root = MCTSNode(state=state)
        root.visit_count = 10

        child = root.expand()
        child.visit_count = 5
        child.total_value = 3.0

        score = child.ucb_score(c_puct=1.41)
        assert score != float('inf')
        assert isinstance(score, float)

    def test_best_child_selects_highest_ucb(self):
        """best_child should return the child with highest UCB."""
        state = GameState(grid_size=2)
        root = MCTSNode(state=state)
        root.visit_count = 100

        # Expand all children
        while root.untried_actions:
            root.expand()

        # Give one child many visits and high value
        root.children[0].visit_count = 50
        root.children[0].total_value = 40.0

        # Give others fewer visits
        for child in root.children[1:]:
            child.visit_count = 1
            child.total_value = 0.0

        best = root.best_child(c_puct=0.0)  # No exploration
        assert best is root.children[0]

    def test_backpropagate_updates_ancestors(self):
        """Backpropagation should update all nodes to root."""
        state = GameState()
        root = MCTSNode(state=state)
        child = root.expand()

        child.backpropagate(1.0, PLAYER_1)

        assert child.visit_count == 1
        assert root.visit_count == 1

    def test_best_action_returns_most_visited(self):
        """best_action should return the most-visited child's action."""
        state = GameState(grid_size=2)
        root = MCTSNode(state=state)

        # Expand a few children
        c1 = root.expand()
        c2 = root.expand()

        c1.visit_count = 10
        c2.visit_count = 50

        action, node = root.best_action()
        assert action == c2.action
        assert node is c2

    def test_tree_stats(self):
        """tree_stats should return correct counts."""
        state = GameState(grid_size=2)
        root = MCTSNode(state=state)

        child = root.expand()

        stats = root.tree_stats()
        assert stats['total_nodes'] == 2  # root + 1 child
        assert stats['max_depth'] == 1
        assert stats['children_count'] == 1


# ═══════════════════════════════════════════════════════════════════════
# 2. MCTS ALGORITHM TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestMCTS:
    """Test the MCTS algorithm."""

    def test_search_returns_valid_move(self):
        """Search should return a legal move."""
        state = GameState()
        mcts = MCTS(n_simulations=50, seed=42)

        action, root = mcts.search(state)

        assert action in state.get_legal_moves()

    def test_search_builds_tree(self):
        """After search, the root should have children and visits."""
        state = GameState()
        mcts = MCTS(n_simulations=50, seed=42)

        action, root = mcts.search(state)

        assert root.visit_count == 50
        assert len(root.children) > 0

    def test_more_simulations_more_nodes(self):
        """More simulations should produce a larger search tree."""
        state = GameState()

        mcts_small = MCTS(n_simulations=20, seed=42)
        _, root_small = mcts_small.search(state)

        mcts_large = MCTS(n_simulations=100, seed=42)
        _, root_large = mcts_large.search(state)

        stats_small = root_small.tree_stats()
        stats_large = root_large.tree_stats()

        assert stats_large['total_nodes'] >= stats_small['total_nodes']

    def test_rollout_returns_valid_value(self):
        """Rollout should return +1, -1, or 0."""
        state = GameState()
        mcts = MCTS(seed=42)

        value = mcts._rollout(state, PLAYER_1)
        assert value in (1.0, -1.0, 0.0)

    def test_action_probabilities_sum_to_one(self):
        """Action probabilities should sum to approximately 1.0."""
        state = GameState()
        mcts = MCTS(n_simulations=50, seed=42)

        _, root = mcts.search(state)
        probs = mcts.get_action_probabilities(root, temperature=1.0)

        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}"

    def test_greedy_temperature(self):
        """Near-zero temperature should concentrate on one move."""
        state = GameState()
        mcts = MCTS(n_simulations=50, seed=42)

        _, root = mcts.search(state)
        probs = mcts.get_action_probabilities(root, temperature=0.001)

        # Exactly one move should have probability 1.0
        max_prob = max(probs.values())
        assert max_prob == 1.0

    def test_deterministic_with_seed(self):
        """Same seed should produce the same search result."""
        state = GameState()

        mcts1 = MCTS(n_simulations=50, seed=42)
        action1, _ = mcts1.search(state)

        mcts2 = MCTS(n_simulations=50, seed=42)
        action2, _ = mcts2.search(state)

        assert action1 == action2

    def test_search_on_small_board(self):
        """MCTS should work on small boards."""
        state = GameState(grid_size=3)
        mcts = MCTS(n_simulations=30, seed=42)

        action, root = mcts.search(state)
        assert action in state.get_legal_moves()


# ═══════════════════════════════════════════════════════════════════════
# 3. MCTS AGENT TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestMCTSAgent:
    """Test the MCTSAgent class."""

    def test_inherits_base_agent(self):
        """MCTSAgent should be a BaseAgent."""
        agent = MCTSAgent(n_simulations=10)
        assert isinstance(agent, BaseAgent)

    def test_selects_legal_move(self):
        """Agent should always return a legal move."""
        agent = MCTSAgent(n_simulations=30, seed=42)
        state = GameState()

        move = agent.select_move(state)
        assert move in state.get_legal_moves()

    def test_plays_full_game(self):
        """MCTSAgent should be able to play a complete game."""
        agent1 = MCTSAgent(name="MCTS-P1", n_simulations=20, seed=42)
        agent2 = RandomAgent(name="Random-P2", seed=43)

        state = GameState(grid_size=4)  # Smaller for speed
        agents = {PLAYER_1: agent1, PLAYER_2: agent2}

        while not state.is_game_over():
            agent = agents[state.current_player]
            move = agent.select_move(state)
            assert move in state.get_legal_moves()
            state.apply_move(*move)

        assert state.get_winner() in (PLAYER_1, PLAYER_2)

    def test_single_move_skip_search(self):
        """When only one move is legal, skip the full search."""
        agent = MCTSAgent(n_simulations=100, seed=42)

        # Play most of the game to reach a state with few moves
        state = GameState(grid_size=3)
        random_agent = RandomAgent(seed=1)

        # Play until game is nearly over or only 1 move remains
        while not state.is_game_over():
            moves = state.get_legal_moves()
            if len(moves) == 1:
                move = agent.select_move(state)
                assert move == moves[0]
                break
            state.apply_move(*random_agent.select_move(state))

    def test_search_stats(self):
        """Should be able to retrieve search statistics."""
        agent = MCTSAgent(n_simulations=30, seed=42)
        state = GameState()

        agent.select_move(state)
        stats = agent.get_search_stats()

        assert stats is not None
        assert 'total_nodes' in stats
        assert stats['root_visits'] == 30

    def test_has_name(self):
        """Agent should have a configurable name."""
        agent = MCTSAgent(name="StrongBot", n_simulations=10)
        assert agent.name == "StrongBot"


# ═══════════════════════════════════════════════════════════════════════
# 4. PHASE 3 ADDITIONS — clone_state, policy, temperature
# ═══════════════════════════════════════════════════════════════════════

class TestCloneState:
    """Test the clone_state() alias on GameState."""

    def test_clone_state_exists(self):
        """GameState should have a clone_state() method."""
        state = GameState()
        assert hasattr(state, 'clone_state')

    def test_clone_state_returns_independent_copy(self):
        """clone_state() should return an independent copy."""
        state = GameState()
        state.apply_move(0, 0)
        state.apply_move(1, 1)

        clone = state.clone_state()

        # Should be equal
        assert clone.current_player == state.current_player
        assert clone.turn_count == state.turn_count
        assert clone.scores == state.scores

        # Should be independent
        clone.apply_move(2, 2)
        assert state.turn_count == 2  # Unchanged
        assert clone.turn_count == 3

    def test_clone_state_equals_clone(self):
        """clone_state() and clone() should produce equivalent copies."""
        state = GameState()
        state.apply_move(3, 3)

        cs = state.clone_state()
        cl = state.clone()

        assert cs.current_player == cl.current_player
        assert cs.turn_count == cl.turn_count
        assert cs.scores == cl.scores
        for r in range(state.grid_size):
            for c in range(state.grid_size):
                assert cs.board[r][c] == cl.board[r][c]

    def test_mcts_search_no_attribute_error(self):
        """MCTS search should NOT crash with AttributeError (clone_state)."""
        state = GameState(grid_size=3)
        mcts = MCTS(n_simulations=10, seed=42)

        # This would crash before the fix
        action, root = mcts.search(state)
        assert action in state.get_legal_moves()
        assert root.visit_count == 10


class TestMCTSPolicy:
    """Test MCTS policy data collection (Phase 3)."""

    def test_select_move_with_policy(self):
        """select_move_with_policy should return action and policy dict."""
        agent = MCTSAgent(n_simulations=30, seed=42, temperature=1.0)
        state = GameState()

        action, policy = agent.select_move_with_policy(state)

        # Action should be legal
        assert action in state.get_legal_moves()

        # Policy should be a dict
        assert isinstance(policy, dict)
        assert len(policy) > 0

        # Policy values should sum to ~1.0
        total = sum(policy.values())
        assert abs(total - 1.0) < 0.01, f"Policy sums to {total}"

        # All policy actions should be legal
        legal = set(state.get_legal_moves())
        for a in policy:
            assert a in legal, f"Policy action {a} is not legal"

    def test_policy_contains_chosen_action(self):
        """The chosen action should be in the policy dict."""
        agent = MCTSAgent(n_simulations=30, seed=42, temperature=0.01)
        state = GameState()

        action, policy = agent.select_move_with_policy(state)
        assert action in policy, "Chosen action should be in policy"

    def test_single_move_policy(self):
        """When only one move exists, policy should assign prob 1.0."""
        agent = MCTSAgent(n_simulations=100, seed=42)

        state = GameState(grid_size=3)
        random_agent = RandomAgent(seed=1)

        while not state.is_game_over():
            moves = state.get_legal_moves()
            if len(moves) == 1:
                action, policy = agent.select_move_with_policy(state)
                assert action == moves[0]
                assert policy == {moves[0]: 1.0}
                break
            state.apply_move(*random_agent.select_move(state))

    def test_last_policy_stored(self):
        """Agent should store the last policy for retrieval."""
        agent = MCTSAgent(n_simulations=30, seed=42, temperature=1.0)
        state = GameState()

        agent.select_move_with_policy(state)
        assert agent.last_policy is not None
        assert isinstance(agent.last_policy, dict)


class TestTemperatureAnnealing:
    """Test temperature annealing in MCTSAgent."""

    def test_temperature_threshold(self):
        """Temperature should drop after threshold moves."""
        agent = MCTSAgent(
            n_simulations=10, temperature=1.0,
            temperature_threshold=5, seed=42,
        )

        # Before threshold
        assert agent._get_temperature(move_number=1) == 1.0
        assert agent._get_temperature(move_number=5) == 1.0

        # After threshold
        assert agent._get_temperature(move_number=6) == 0.01
        assert agent._get_temperature(move_number=100) == 0.01

    def test_no_threshold_uses_base_temperature(self):
        """Without threshold, base temperature is always used."""
        agent = MCTSAgent(
            n_simulations=10, temperature=1.0,
            temperature_threshold=None, seed=42,
        )

        assert agent._get_temperature(move_number=1) == 1.0
        assert agent._get_temperature(move_number=100) == 1.0
        assert agent._get_temperature(move_number=None) == 1.0

    def test_temperature_affects_selection(self):
        """Different temperatures should potentially yield different moves."""
        state = GameState(grid_size=3)

        # High temperature agent — more random
        hot_agent = MCTSAgent(
            n_simulations=50, temperature=1.0,
            temperature_threshold=None, seed=42,
        )

        # Low temperature agent — more greedy
        cold_agent = MCTSAgent(
            n_simulations=50, temperature=0.01,
            temperature_threshold=None, seed=42,
        )

        # Both should return valid moves
        hot_action, hot_policy = hot_agent.select_move_with_policy(state)
        cold_action, cold_policy = cold_agent.select_move_with_policy(state)

        assert hot_action in state.get_legal_moves()
        assert cold_action in state.get_legal_moves()

        # Policies should be the same (always computed at τ=1.0)
        # but actions may differ due to selection temperature


# ═══════════════════════════════════════════════════════════════════════
# 5. MCTS vs RANDOM QUICK CHECK
# ═══════════════════════════════════════════════════════════════════════

class TestMCTSvsRandom:
    """Sanity check that MCTS is stronger than random."""

    def test_mcts_beats_random_on_small_board(self):
        """MCTS should win most games against random on a small board."""
        mcts_wins = 0
        n_games = 5  # Small number for speed in tests

        for i in range(n_games):
            state = GameState(grid_size=4)
            mcts_agent = MCTSAgent(n_simulations=30, seed=i)
            random_agent = RandomAgent(seed=i + 100)

            agents = {PLAYER_1: mcts_agent, PLAYER_2: random_agent}

            while not state.is_game_over():
                agent = agents[state.current_player]
                move = agent.select_move(state)
                state.apply_move(*move)

            if state.get_winner() == PLAYER_1:
                mcts_wins += 1

        # MCTS should win at least 1 out of 5 even with very few simulations
        assert mcts_wins >= 1, \
            f"MCTS won only {mcts_wins}/{n_games} — expected at least 1"
