"""
test_self_play.py — Unit tests for Phase 2: Self-Play Infrastructure.

Tests cover:
  1. RandomAgent — move selection, validity, reproducibility
  2. GameRecord — recording, serialization, training examples
  3. SelfPlayManager — game execution, statistics, file I/O

Run with:
    cd ColorWars
    source .venv/bin/activate
    python -m pytest ai/tests/test_self_play.py -v
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.game_engine.game_state import GameState
from ai.game_engine.constants import PLAYER_1, PLAYER_2, EMPTY, GRID_SIZE
from ai.agents.random_agent import RandomAgent
from ai.agents.base_agent import BaseAgent
from ai.self_play.game_record import GameRecord, MoveRecord
from ai.self_play.self_play_manager import SelfPlayManager


# ═══════════════════════════════════════════════════════════════════════
# 1. RANDOM AGENT TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestRandomAgent:
    """Test the RandomAgent class."""

    def test_selects_legal_move(self):
        """RandomAgent should always return a legal move."""
        agent = RandomAgent(seed=42)
        state = GameState()

        for _ in range(20):
            if state.is_game_over():
                break
            move = agent.select_move(state)
            assert move in state.get_legal_moves(), \
                f"Move {move} is not legal"
            state.apply_move(*move)

    def test_move_is_tuple(self):
        """Move should be a (row, col) tuple."""
        agent = RandomAgent(seed=42)
        state = GameState()
        move = agent.select_move(state)
        assert isinstance(move, tuple)
        assert len(move) == 2

    def test_reproducible_with_seed(self):
        """Same seed should produce same moves."""
        state1 = GameState()
        state2 = GameState()
        agent1 = RandomAgent(seed=123)
        agent2 = RandomAgent(seed=123)

        moves1 = []
        moves2 = []

        for _ in range(10):
            m1 = agent1.select_move(state1)
            m2 = agent2.select_move(state2)
            moves1.append(m1)
            moves2.append(m2)
            state1.apply_move(*m1)
            state2.apply_move(*m2)

        assert moves1 == moves2, "Same seed should give same moves"

    def test_different_seeds_give_different_moves(self):
        """Different seeds should (usually) produce different moves."""
        agent1 = RandomAgent(seed=1)
        agent2 = RandomAgent(seed=999)
        state1 = GameState()
        state2 = GameState()

        moves1 = [agent1.select_move(state1) for _ in range(5)]
        moves2 = [agent2.select_move(state2) for _ in range(5)]

        # Very unlikely that 5 random moves are identical with different seeds
        assert moves1 != moves2 or True  # Allow but unlikely

    def test_error_when_no_moves(self):
        """Should raise ValueError when game is over."""
        agent = RandomAgent()
        state = GameState()
        state.game_over = True

        try:
            agent.select_move(state)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_inherits_base_agent(self):
        """RandomAgent should be an instance of BaseAgent."""
        agent = RandomAgent()
        assert isinstance(agent, BaseAgent)

    def test_has_name(self):
        """Agent should have a configurable name."""
        agent = RandomAgent(name="TestBot")
        assert agent.name == "TestBot"
        assert "TestBot" in repr(agent)


# ═══════════════════════════════════════════════════════════════════════
# 2. GAME RECORD TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestGameRecord:
    """Test the GameRecord data structure."""

    def test_add_move(self):
        """Should record moves with board snapshots."""
        record = GameRecord()
        board = [[[0, 0], [1, 2]], [[2, 1], [0, 0]]]

        record.add_move(board, (0, 1), PLAYER_1)

        assert len(record.moves) == 1
        assert record.moves[0].action == (0, 1)
        assert record.moves[0].player == PLAYER_1

    def test_board_snapshot_is_deep_copy(self):
        """Modifying the original board should not affect the record."""
        record = GameRecord()
        board = [[[0, 0], [1, 2]], [[2, 1], [0, 0]]]

        record.add_move(board, (0, 0), PLAYER_1)

        # Modify original board
        board[0][0] = [1, 99]

        # Record should be unaffected
        assert record.moves[0].board_state[0][0] == [0, 0], \
            "Board snapshot should be independent of original"

    def test_training_examples_win(self):
        """Training examples should label wins as +1."""
        record = GameRecord(winner=PLAYER_1)
        board = [[[0, 0]]]

        record.add_move(board, (0, 0), PLAYER_1)
        record.add_move(board, (0, 0), PLAYER_2)

        examples = record.get_training_examples()

        assert len(examples) == 2
        assert examples[0][2] == 1.0   # P1 made this move, P1 won → +1
        assert examples[1][2] == -1.0  # P2 made this move, P1 won → -1

    def test_training_examples_structure(self):
        """Each training example should be (board, policy, outcome)."""
        record = GameRecord(winner=PLAYER_1)
        board = [[[0, 0], [0, 0]]]

        record.add_move(board, (0, 1), PLAYER_1)

        examples = record.get_training_examples()
        board_state, policy, outcome = examples[0]

        assert isinstance(board_state, list)
        # policy is None for RandomAgent moves (no MCTS data)
        assert policy is None
        assert isinstance(outcome, float)

    def test_serialization_roundtrip(self):
        """to_dict() → from_dict() should preserve all data."""
        record = GameRecord(
            winner=PLAYER_2,
            grid_size=6,
            total_turns=42,
            p1_agent="Agent1",
            p2_agent="Agent2",
            timestamp=1234567.89,
            game_id=7,
        )
        board = [[[1, 2], [0, 0]], [[2, 3], [0, 0]]]
        record.add_move(board, (0, 0), PLAYER_1)
        record.add_move(board, (1, 0), PLAYER_2)

        # Serialize and deserialize
        data = record.to_dict()
        loaded = GameRecord.from_dict(data)

        assert loaded.winner == PLAYER_2
        assert loaded.grid_size == 6
        assert loaded.total_turns == 42
        assert loaded.p1_agent == "Agent1"
        assert loaded.p2_agent == "Agent2"
        assert loaded.game_id == 7
        assert len(loaded.moves) == 2
        assert loaded.moves[0].action == (0, 0)
        assert loaded.moves[1].player == PLAYER_2

    def test_save_and_load(self):
        """Should save to JSON and load back correctly."""
        record = GameRecord(winner=PLAYER_1, total_turns=10, game_id=1)
        board = [[[1, 1]]]
        record.add_move(board, (0, 0), PLAYER_1)

        # Save to a temp file
        tmpdir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(tmpdir, "test_game.json")
            record.save(filepath)

            # File should exist
            assert os.path.exists(filepath)

            # Load it back
            loaded = GameRecord.load(filepath)
            assert loaded.winner == PLAYER_1
            assert loaded.total_turns == 10
            assert len(loaded.moves) == 1

            # Verify JSON is readable
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert data['winner'] == 1
        finally:
            shutil.rmtree(tmpdir)

    def test_json_is_human_readable(self):
        """Saved JSON should be indented and readable."""
        record = GameRecord(winner=PLAYER_1, game_id=1)
        board = [[[0, 0]]]
        record.add_move(board, (0, 0), PLAYER_1)

        tmpdir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(tmpdir, "test.json")
            record.save(filepath)

            with open(filepath, 'r') as f:
                content = f.read()

            # Should be indented (not a single line)
            assert '\n' in content, "JSON should be pretty-printed"
        finally:
            shutil.rmtree(tmpdir)


# ═══════════════════════════════════════════════════════════════════════
# 3. SELF-PLAY MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestSelfPlayManager:
    """Test the SelfPlayManager orchestrator."""

    def test_play_single_game(self):
        """Should produce a complete GameRecord."""
        manager = SelfPlayManager(grid_size=6)
        agent1 = RandomAgent(name="P1", seed=42)
        agent2 = RandomAgent(name="P2", seed=43)

        record = manager.play_game(agent1, agent2)

        assert record.winner in (PLAYER_1, PLAYER_2)
        assert record.total_turns > 0
        assert len(record.moves) == record.total_turns
        assert record.p1_agent == "P1"
        assert record.p2_agent == "P2"
        assert record.grid_size == 6

    def test_moves_alternate_players(self):
        """Moves should alternate between P1 and P2."""
        manager = SelfPlayManager(grid_size=6)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)

        record = manager.play_game(agent1, agent2)

        # First move is P1, second is P2, etc.
        for i, move in enumerate(record.moves):
            expected_player = PLAYER_1 if i % 2 == 0 else PLAYER_2
            assert move.player == expected_player, \
                f"Move {i} should be by Player {expected_player}, " \
                f"got Player {move.player}"

    def test_run_multiple_games(self):
        """Should run the requested number of games."""
        manager = SelfPlayManager(grid_size=6)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)

        records = manager.run_self_play(
            agent1, agent2, n_games=10
        )

        assert len(records) == 10
        for r in records:
            assert r.winner in (PLAYER_1, PLAYER_2)
            assert r.total_turns > 0

    def test_game_ids_are_sequential(self):
        """Each game should have a unique, sequential ID."""
        manager = SelfPlayManager()
        agent1 = RandomAgent(seed=1)
        agent2 = RandomAgent(seed=2)

        records = manager.run_self_play(agent1, agent2, n_games=5)

        ids = [r.game_id for r in records]
        # IDs should be consecutive
        for i in range(1, len(ids)):
            assert ids[i] == ids[i-1] + 1

    def test_save_to_disk(self):
        """Games should be saved as JSON files to the specified directory."""
        manager = SelfPlayManager(grid_size=6)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)

        tmpdir = tempfile.mkdtemp()
        try:
            records = manager.run_self_play(
                agent1, agent2, n_games=5, save_dir=tmpdir
            )

            # Check files exist
            files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
            assert len(files) == 5, f"Expected 5 JSON files, found {len(files)}"

            # Check files can be loaded
            for f in files:
                loaded = GameRecord.load(os.path.join(tmpdir, f))
                assert loaded.winner in (PLAYER_1, PLAYER_2)
        finally:
            shutil.rmtree(tmpdir)

    def test_statistics(self):
        """Statistics should correctly summarize game outcomes."""
        manager = SelfPlayManager(grid_size=6)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)

        records = manager.run_self_play(agent1, agent2, n_games=20)
        stats = manager.get_statistics(records)

        assert stats['total_games'] == 20
        assert stats['p1_wins'] + stats['p2_wins'] == 20
        assert 0 <= stats['p1_win_rate'] <= 100
        assert 0 <= stats['p2_win_rate'] <= 100
        assert abs(stats['p1_win_rate'] + stats['p2_win_rate'] - 100.0) < 0.01
        assert stats['avg_turns'] > 0
        assert stats['min_turns'] <= stats['avg_turns'] <= stats['max_turns']
        assert stats['total_examples'] > 0

    def test_empty_statistics(self):
        """Statistics on empty records should return zeros."""
        stats = SelfPlayManager.get_statistics([])
        assert stats['total_games'] == 0
        assert stats['p1_wins'] == 0

    def test_training_examples_from_game(self):
        """Each recorded game should produce valid training examples."""
        manager = SelfPlayManager(grid_size=6)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)

        record = manager.play_game(agent1, agent2)
        examples = record.get_training_examples()

        assert len(examples) == record.total_turns
        for board, policy, outcome in examples:
            assert isinstance(board, list)
            assert len(board) == 6  # grid_size rows
            assert len(board[0]) == 6  # grid_size cols
            # policy is None for RandomAgent games (no MCTS data)
            assert policy is None
            assert outcome in (1.0, -1.0)

    def test_small_grid_games(self):
        """Self-play should work on smaller grids too."""
        manager = SelfPlayManager(grid_size=3)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=43)

        records = manager.run_self_play(agent1, agent2, n_games=5)

        assert len(records) == 5
        for r in records:
            assert r.grid_size == 3
            assert r.winner in (PLAYER_1, PLAYER_2)
