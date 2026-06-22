"""
test_training_pipeline.py — Unit tests for Phase 3 training pipeline.

Tests cover:
  1. Board encoding — 5-plane representation, perspective encoding
  2. Policy encoding/decoding — roundtrip consistency
  3. Data augmentation — 8-fold symmetry transformations
  4. GameRecord with policy — serialization, training examples
  5. Training pipeline — end-to-end data generation
  6. Self-play manager — MCTS training games

Run with:
    cd ColorWars
    source .venv/bin/activate
    python -m pytest ai/tests/test_training_pipeline.py -v
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.game_engine.game_state import GameState
from ai.game_engine.constants import PLAYER_1, PLAYER_2, EMPTY, GRID_SIZE
from ai.self_play.game_record import GameRecord, MoveRecord
from ai.self_play.board_encoder import (
    encode_board,
    encode_board_from_perspective,
    encode_policy,
    decode_policy,
    augment_training_example,
    _transform_action,
    _transform_board,
    _transform_policy,
)
from ai.self_play.training_pipeline import TrainingPipeline
from ai.self_play.self_play_manager import SelfPlayManager
from ai.agents.mcts_agent import MCTSAgent
from ai.agents.random_agent import RandomAgent


# ═══════════════════════════════════════════════════════════════════════
# 1. BOARD ENCODING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestBoardEncoding:
    """Test encode_board() and encode_board_from_perspective()."""

    def test_empty_board_encoding(self):
        """Empty board should have all zeros except current player plane."""
        board = [[[EMPTY, 0] for _ in range(4)] for _ in range(4)]
        planes = encode_board(board, PLAYER_1, grid_size=4)

        assert len(planes) == 5

        # P1 ownership — all zeros
        for r in range(4):
            for c in range(4):
                assert planes[0][r][c] == 0.0

        # P2 ownership — all zeros
        for r in range(4):
            for c in range(4):
                assert planes[2][r][c] == 0.0

        # Current player plane — all 1s (P1 to move)
        for r in range(4):
            for c in range(4):
                assert planes[4][r][c] == 1.0

    def test_p2_to_move_plane(self):
        """When P2 is to move, current player plane should be all 0s."""
        board = [[[EMPTY, 0] for _ in range(3)] for _ in range(3)]
        planes = encode_board(board, PLAYER_2, grid_size=3)

        for r in range(3):
            for c in range(3):
                assert planes[4][r][c] == 0.0

    def test_ownership_encoding(self):
        """P1 and P2 cells should appear in correct planes."""
        board = [
            [[PLAYER_1, 2], [EMPTY, 0], [PLAYER_2, 3]],
            [[EMPTY, 0],    [PLAYER_1, 1], [EMPTY, 0]],
            [[PLAYER_2, 1], [EMPTY, 0],    [EMPTY, 0]],
        ]
        planes = encode_board(board, PLAYER_1, grid_size=3)

        # P1 ownership plane
        assert planes[0][0][0] == 1.0  # P1 at (0,0)
        assert planes[0][1][1] == 1.0  # P1 at (1,1)
        assert planes[0][0][2] == 0.0  # P2 at (0,2) — not P1

        # P2 ownership plane
        assert planes[2][0][2] == 1.0  # P2 at (0,2)
        assert planes[2][2][0] == 1.0  # P2 at (2,0)
        assert planes[2][0][0] == 0.0  # P1 at (0,0) — not P2

    def test_dot_normalization(self):
        """Dots should be normalized by EXPLOSION_THRESHOLD."""
        from ai.game_engine.constants import EXPLOSION_THRESHOLD

        board = [[[PLAYER_1, 3], [PLAYER_2, 2]],
                 [[EMPTY, 0],    [PLAYER_1, 1]]]
        planes = encode_board(board, PLAYER_1, grid_size=2)

        # P1 dots: 3/4 = 0.75, 1/4 = 0.25
        assert abs(planes[1][0][0] - 3 / EXPLOSION_THRESHOLD) < 0.001
        assert abs(planes[1][1][1] - 1 / EXPLOSION_THRESHOLD) < 0.001

        # P2 dots: 2/4 = 0.5
        assert abs(planes[3][0][1] - 2 / EXPLOSION_THRESHOLD) < 0.001

    def test_perspective_encoding(self):
        """Perspective encoding should swap P1/P2 based on current player."""
        board = [
            [[PLAYER_1, 2], [PLAYER_2, 1]],
            [[EMPTY, 0],    [PLAYER_1, 3]],
        ]

        # From P1's perspective: P1 = "me", P2 = "opponent"
        planes_p1 = encode_board_from_perspective(board, PLAYER_1, grid_size=2)
        assert planes_p1[0][0][0] == 1.0  # "me" = P1 at (0,0)
        assert planes_p1[2][0][1] == 1.0  # "opp" = P2 at (0,1)

        # From P2's perspective: P2 = "me", P1 = "opponent"
        planes_p2 = encode_board_from_perspective(board, PLAYER_2, grid_size=2)
        assert planes_p2[0][0][1] == 1.0  # "me" = P2 at (0,1)
        assert planes_p2[2][0][0] == 1.0  # "opp" = P1 at (0,0)

    def test_encoding_shape(self):
        """Encoding should produce 5 planes of correct dimensions."""
        board = [[[EMPTY, 0] for _ in range(6)] for _ in range(6)]
        planes = encode_board(board, PLAYER_1, grid_size=6)

        assert len(planes) == 5
        for plane in planes:
            assert len(plane) == 6
            for row in plane:
                assert len(row) == 6


# ═══════════════════════════════════════════════════════════════════════
# 2. POLICY ENCODING / DECODING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestPolicyEncoding:
    """Test policy encoding and decoding."""

    def test_encode_policy_basic(self):
        """Policy should encode to flat vector of correct length."""
        policy = {(0, 0): 0.4, (1, 1): 0.3, (2, 2): 0.3}
        flat = encode_policy(policy, grid_size=3)

        assert len(flat) == 9  # 3×3
        assert abs(flat[0] - 0.4) < 0.001   # (0,0) → index 0
        assert abs(flat[4] - 0.3) < 0.001   # (1,1) → index 4
        assert abs(flat[8] - 0.3) < 0.001   # (2,2) → index 8

    def test_encode_none_policy(self):
        """None policy should produce uniform distribution."""
        flat = encode_policy(None, grid_size=3)
        assert len(flat) == 9
        expected = 1.0 / 9
        for v in flat:
            assert abs(v - expected) < 0.001

    def test_policy_roundtrip(self):
        """encode → decode should preserve the policy."""
        policy = {(0, 1): 0.5, (2, 0): 0.3, (1, 2): 0.2}
        flat = encode_policy(policy, grid_size=3)
        decoded = decode_policy(flat, grid_size=3)

        for action, prob in policy.items():
            assert action in decoded
            assert abs(decoded[action] - prob) < 0.001

    def test_decode_skips_zeros(self):
        """Decoded policy should not include zero-probability actions."""
        flat = [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        decoded = decode_policy(flat, grid_size=3)
        assert len(decoded) == 2
        assert (0, 1) in decoded
        assert (1, 0) in decoded

    def test_encode_policy_sums_correctly(self):
        """Encoded policy should sum to the same as input."""
        policy = {(0, 0): 0.2, (0, 1): 0.3, (1, 0): 0.5}
        flat = encode_policy(policy, grid_size=2)
        assert abs(sum(flat) - 1.0) < 0.001


# ═══════════════════════════════════════════════════════════════════════
# 3. DATA AUGMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDataAugmentation:
    """Test 8-fold symmetry data augmentation."""

    def test_identity_transform(self):
        """Transform 0 (identity) should not change coordinates."""
        assert _transform_action((1, 2), 0, grid_size=4) == (1, 2)
        assert _transform_action((0, 0), 0, grid_size=6) == (0, 0)

    def test_rotate_90(self):
        """Transform 1 (rotate 90° CW) should map correctly."""
        # On a 4×4 grid: (0,0) → (0,3), (0,3) → (3,3), (3,3) → (3,0), (3,0) → (0,0)
        n = 4
        assert _transform_action((0, 0), 1, n) == (0, 3)
        assert _transform_action((0, 3), 1, n) == (3, 3)
        assert _transform_action((3, 3), 1, n) == (3, 0)
        assert _transform_action((3, 0), 1, n) == (0, 0)

    def test_rotate_180(self):
        """Transform 2 (rotate 180°) should map correctly."""
        n = 4
        assert _transform_action((0, 0), 2, n) == (3, 3)
        assert _transform_action((3, 3), 2, n) == (0, 0)
        assert _transform_action((1, 2), 2, n) == (2, 1)

    def test_all_transforms_produce_valid_coords(self):
        """All 8 transforms should produce in-bounds coordinates."""
        n = 6
        for r in range(n):
            for c in range(n):
                for t in range(8):
                    new_r, new_c = _transform_action((r, c), t, n)
                    assert 0 <= new_r < n, \
                        f"Transform {t}: ({r},{c}) → ({new_r},{new_c}) out of bounds"
                    assert 0 <= new_c < n, \
                        f"Transform {t}: ({r},{c}) → ({new_r},{new_c}) out of bounds"

    def test_transforms_are_bijections(self):
        """Each transform should be a bijection (no two inputs map to same output)."""
        n = 4
        for t in range(8):
            outputs = set()
            for r in range(n):
                for c in range(n):
                    out = _transform_action((r, c), t, n)
                    assert out not in outputs, \
                        f"Transform {t}: collision at output {out}"
                    outputs.add(out)
            assert len(outputs) == n * n

    def test_augment_produces_8_examples(self):
        """augment_training_example should produce exactly 8 examples."""
        board = [[[PLAYER_1, 1], [EMPTY, 0]],
                 [[EMPTY, 0],    [PLAYER_2, 2]]]
        policy = {(0, 0): 0.6, (0, 1): 0.4}
        outcome = 1.0

        augmented = augment_training_example(board, policy, outcome,
                                             grid_size=2)

        assert len(augmented) == 8

        # All should have the same outcome
        for _, _, o in augmented:
            assert o == 1.0

        # All policies should sum to ~1.0
        for _, p, _ in augmented:
            if p is not None:
                assert abs(sum(p.values()) - 1.0) < 0.001

    def test_augment_with_none_policy(self):
        """Augmentation should handle None policy gracefully."""
        board = [[[PLAYER_1, 1], [EMPTY, 0]],
                 [[EMPTY, 0],    [PLAYER_2, 2]]]

        augmented = augment_training_example(board, None, -1.0, grid_size=2)

        assert len(augmented) == 8
        for _, p, _ in augmented:
            assert p is None

    def test_board_transform_preserves_content(self):
        """Board transformation should preserve cell values."""
        board = [
            [[PLAYER_1, 3], [PLAYER_2, 1]],
            [[EMPTY, 0],    [PLAYER_1, 2]],
        ]

        for t in range(8):
            transformed = _transform_board(board, t, grid_size=2)
            # Count cells — should be same
            flat_orig = []
            flat_trans = []
            for r in range(2):
                for c in range(2):
                    flat_orig.append(tuple(board[r][c]))
                    flat_trans.append(tuple(transformed[r][c]))
            assert sorted(flat_orig) == sorted(flat_trans), \
                f"Transform {t} changed cell values"

    def test_policy_transform_preserves_probabilities(self):
        """Policy transformation should preserve probability values."""
        policy = {(0, 0): 0.5, (0, 1): 0.3, (1, 0): 0.2}

        for t in range(8):
            transformed = _transform_policy(policy, t, grid_size=2)
            assert abs(sum(transformed.values()) - 1.0) < 0.001
            assert sorted(transformed.values()) == sorted(policy.values())


# ═══════════════════════════════════════════════════════════════════════
# 4. GAME RECORD WITH POLICY TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestGameRecordWithPolicy:
    """Test GameRecord with MCTS policy data."""

    def test_add_move_with_policy(self):
        """Should record moves with policy data."""
        record = GameRecord()
        board = [[[0, 0], [1, 2]], [[2, 1], [0, 0]]]
        policy = {(0, 0): 0.6, (0, 1): 0.4}

        record.add_move(board, (0, 0), PLAYER_1, action_probs=policy)

        assert len(record.moves) == 1
        assert record.moves[0].action_probabilities is not None
        assert record.moves[0].action_probabilities[(0, 0)] == 0.6
        assert record.has_policy is True

    def test_add_move_without_policy(self):
        """Should record moves without policy (backward compatible)."""
        record = GameRecord()
        board = [[[0, 0], [1, 2]], [[2, 1], [0, 0]]]

        record.add_move(board, (0, 0), PLAYER_1)

        assert record.moves[0].action_probabilities is None
        assert record.has_policy is False

    def test_training_examples_with_policy(self):
        """Training examples should include policy data."""
        record = GameRecord(winner=PLAYER_1)
        board = [[[0, 0]]]
        policy = {(0, 0): 1.0}

        record.add_move(board, (0, 0), PLAYER_1, action_probs=policy)
        record.add_move(board, (0, 0), PLAYER_2)

        examples = record.get_training_examples()
        assert len(examples) == 2

        # First move has policy
        _, p1, o1 = examples[0]
        assert p1 is not None
        assert p1[(0, 0)] == 1.0
        assert o1 == 1.0

        # Second move has no policy
        _, p2, o2 = examples[1]
        assert p2 is None
        assert o2 == -1.0

    def test_serialization_with_policy(self):
        """Serialization should preserve policy data."""
        record = GameRecord(
            winner=PLAYER_1, grid_size=4,
            total_turns=2, game_id=1,
        )
        board = [[[0, 0] for _ in range(4)] for _ in range(4)]
        policy = {(0, 0): 0.5, (1, 1): 0.3, (2, 2): 0.2}

        record.add_move(board, (0, 0), PLAYER_1, action_probs=policy)

        # Serialize and deserialize
        data = record.to_dict()
        loaded = GameRecord.from_dict(data)

        assert loaded.has_policy is True
        assert len(loaded.moves) == 1
        probs = loaded.moves[0].action_probabilities
        assert probs is not None
        assert abs(probs[(0, 0)] - 0.5) < 0.001
        assert abs(probs[(1, 1)] - 0.3) < 0.001
        assert abs(probs[(2, 2)] - 0.2) < 0.001

    def test_save_load_with_policy(self):
        """Should save and load games with policy data."""
        record = GameRecord(
            winner=PLAYER_2, total_turns=5, game_id=1,
        )
        board = [[[0, 0]]]
        policy = {(0, 0): 1.0}
        record.add_move(board, (0, 0), PLAYER_1, action_probs=policy)

        tmpdir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(tmpdir, "test_game.json")
            record.save(filepath)

            loaded = GameRecord.load(filepath)
            assert loaded.has_policy is True
            assert loaded.moves[0].action_probabilities[(0, 0)] == 1.0
        finally:
            shutil.rmtree(tmpdir)

    def test_backward_compatible_load(self):
        """Should load old-format games (no policy data) gracefully."""
        old_data = {
            'game_id': 1,
            'winner': 1,
            'grid_size': 2,
            'total_turns': 1,
            'p1_agent': 'TestP1',
            'p2_agent': 'TestP2',
            'timestamp': 0.0,
            'moves': [
                {
                    'board_state': [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                    'action': [0, 0],
                    'player': 1,
                    # Note: no 'action_probabilities' key
                }
            ],
        }

        record = GameRecord.from_dict(old_data)
        assert record.has_policy is False
        assert record.moves[0].action_probabilities is None


# ═══════════════════════════════════════════════════════════════════════
# 5. TRAINING PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestTrainingPipeline:
    """Test the TrainingPipeline end-to-end."""

    def test_extract_examples_basic(self):
        """Should extract examples from game records."""
        record = GameRecord(winner=PLAYER_1, grid_size=3)
        board = [[[0, 0] for _ in range(3)] for _ in range(3)]
        policy = {(0, 0): 0.5, (1, 1): 0.5}
        record.add_move(board, (0, 0), PLAYER_1, action_probs=policy)
        record.add_move(board, (1, 1), PLAYER_2)

        pipeline = TrainingPipeline(grid_size=3)
        examples = pipeline.extract_examples([record])

        assert len(examples) == 2

    def test_extract_with_augmentation(self):
        """Augmentation should multiply examples by 8×."""
        record = GameRecord(winner=PLAYER_1, grid_size=3)
        board = [[[0, 0] for _ in range(3)] for _ in range(3)]
        policy = {(0, 0): 1.0}
        record.add_move(board, (0, 0), PLAYER_1, action_probs=policy)

        pipeline = TrainingPipeline(grid_size=3)

        # Without augmentation
        raw = pipeline.extract_examples([record], augment=False)
        assert len(raw) == 1

        # With augmentation — moves with policy get 8× augmentation
        aug = pipeline.extract_examples([record], augment=True)
        assert len(aug) == 8

    def test_extract_policy_only(self):
        """policy_only should filter out non-MCTS moves."""
        record = GameRecord(winner=PLAYER_1, grid_size=3)
        board = [[[0, 0] for _ in range(3)] for _ in range(3)]

        record.add_move(board, (0, 0), PLAYER_1,
                        action_probs={(0, 0): 1.0})
        record.add_move(board, (1, 1), PLAYER_2)  # No policy

        pipeline = TrainingPipeline(grid_size=3)
        examples = pipeline.extract_examples([record], policy_only=True)

        assert len(examples) == 1  # Only the MCTS move

    def test_encode_examples(self):
        """Encoded examples should have correct format."""
        record = GameRecord(winner=PLAYER_1, grid_size=3)
        board = [[[PLAYER_1, 1], [EMPTY, 0], [EMPTY, 0]],
                 [[EMPTY, 0],    [EMPTY, 0], [EMPTY, 0]],
                 [[EMPTY, 0],    [EMPTY, 0], [EMPTY, 0]]]
        policy = {(0, 0): 0.7, (1, 1): 0.3}
        record.add_move(board, (0, 0), PLAYER_1, action_probs=policy)

        pipeline = TrainingPipeline(grid_size=3)
        raw = pipeline.extract_examples([record])
        encoded = pipeline.encode_examples(raw)

        assert len(encoded) == 1
        planes, flat_policy, value = encoded[0]

        # 5 planes
        assert len(planes) == 5
        # Each plane is 3×3
        assert len(planes[0]) == 3
        assert len(planes[0][0]) == 3
        # Flat policy is length 9
        assert len(flat_policy) == 9
        # Value is float
        assert isinstance(value, float)

    def test_sample_batch(self):
        """Should sample a random batch of the correct size."""
        examples = [(i, None, 1.0) for i in range(100)]

        pipeline = TrainingPipeline()
        batch = pipeline.sample_batch(examples, batch_size=10)

        assert len(batch) == 10

    def test_sample_batch_larger_than_data(self):
        """If batch_size > data, return all data shuffled."""
        examples = [(i, None, 1.0) for i in range(5)]

        pipeline = TrainingPipeline()
        batch = pipeline.sample_batch(examples, batch_size=100)

        assert len(batch) == 5

    def test_data_stats(self):
        """Should compute correct statistics."""
        examples = [
            (None, {(0, 0): 1.0}, 1.0),
            (None, {(0, 0): 1.0}, -1.0),
            (None, None, 1.0),
            (None, None, -1.0),
        ]

        stats = TrainingPipeline.get_data_stats(examples)
        assert stats['total_examples'] == 4
        assert stats['with_policy'] == 2
        assert stats['without_policy'] == 2
        assert stats['positive_outcomes'] == 2
        assert stats['negative_outcomes'] == 2
        assert abs(stats['outcome_balance'] - 1.0) < 0.001

    def test_load_games_from_directory(self):
        """Should load saved games from a directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            # Save some test games
            for i in range(3):
                record = GameRecord(
                    winner=PLAYER_1, grid_size=3,
                    total_turns=5, game_id=i + 1,
                )
                board = [[[0, 0] for _ in range(3)] for _ in range(3)]
                record.add_move(board, (0, 0), PLAYER_1)
                record.save(os.path.join(tmpdir, f"game_{i + 1:06d}.json"))

            pipeline = TrainingPipeline(grid_size=3)
            records = pipeline.load_games(tmpdir)

            assert len(records) == 3
        finally:
            shutil.rmtree(tmpdir)


# ═══════════════════════════════════════════════════════════════════════
# 6. SELF-PLAY MANAGER TRAINING GAME TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestSelfPlayManagerTraining:
    """Test SelfPlayManager training game functionality."""

    def test_play_training_game_with_mcts(self):
        """Training game with MCTS agents should produce policy data."""
        manager = SelfPlayManager(grid_size=4)
        agent1 = MCTSAgent(name="MCTS-1", n_simulations=10,
                           temperature=1.0, seed=42)
        agent2 = MCTSAgent(name="MCTS-2", n_simulations=10,
                           temperature=1.0, seed=43)

        record = manager.play_training_game(agent1, agent2)

        assert record.winner in (PLAYER_1, PLAYER_2)
        assert record.total_turns > 0
        assert record.has_policy is True

        # Check that moves have policy data
        for move in record.moves:
            assert move.action_probabilities is not None
            assert len(move.action_probabilities) > 0
            total = sum(move.action_probabilities.values())
            assert abs(total - 1.0) < 0.02, \
                f"Policy sums to {total}"

    def test_play_training_game_mixed_agents(self):
        """Training game with mixed agents (MCTS + Random)."""
        manager = SelfPlayManager(grid_size=4)
        mcts_agent = MCTSAgent(name="MCTS", n_simulations=10, seed=42)
        random_agent = RandomAgent(name="Random", seed=43)

        record = manager.play_training_game(mcts_agent, random_agent)

        assert record.winner in (PLAYER_1, PLAYER_2)
        assert record.has_policy is True  # At least MCTS moves have policy

        # MCTS moves should have policy, random moves should not
        mcts_moves = [m for m in record.moves
                      if m.action_probabilities is not None]
        random_moves = [m for m in record.moves
                        if m.action_probabilities is None]
        assert len(mcts_moves) > 0
        assert len(random_moves) > 0

    def test_run_training_self_play(self):
        """Should run multiple training games."""
        manager = SelfPlayManager(grid_size=3)
        agent1 = MCTSAgent(name="A", n_simulations=10, seed=42)
        agent2 = MCTSAgent(name="B", n_simulations=10, seed=43)

        records = manager.run_training_self_play(
            agent1, agent2, n_games=3, swap_colors=True,
        )

        assert len(records) == 3
        for r in records:
            assert r.winner in (PLAYER_1, PLAYER_2)
            assert r.has_policy is True

    def test_color_swapping(self):
        """Color swapping should alternate agent names in records."""
        manager = SelfPlayManager(grid_size=3)
        agent1 = MCTSAgent(name="Alpha", n_simulations=10, seed=42)
        agent2 = MCTSAgent(name="Beta", n_simulations=10, seed=43)

        records = manager.run_training_self_play(
            agent1, agent2, n_games=4, swap_colors=True,
        )

        # Games 1, 3: Alpha=P1, Beta=P2
        assert records[0].p1_agent == "Alpha"
        assert records[0].p2_agent == "Beta"
        assert records[2].p1_agent == "Alpha"
        assert records[2].p2_agent == "Beta"

        # Games 2, 4: Beta=P1, Alpha=P2
        assert records[1].p1_agent == "Beta"
        assert records[1].p2_agent == "Alpha"
        assert records[3].p1_agent == "Beta"
        assert records[3].p2_agent == "Alpha"

    def test_training_save_to_disk(self):
        """Training games should be saveable to disk."""
        manager = SelfPlayManager(grid_size=3)
        agent1 = MCTSAgent(name="A", n_simulations=10, seed=42)
        agent2 = MCTSAgent(name="B", n_simulations=10, seed=43)

        tmpdir = tempfile.mkdtemp()
        try:
            records = manager.run_training_self_play(
                agent1, agent2, n_games=2, save_dir=tmpdir,
            )

            files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
            assert len(files) == 2

            # Load and verify
            loaded = GameRecord.load(os.path.join(tmpdir, sorted(files)[0]))
            assert loaded.has_policy is True
        finally:
            shutil.rmtree(tmpdir)
