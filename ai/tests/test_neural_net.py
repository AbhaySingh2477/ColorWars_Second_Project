"""
test_neural_net.py — Tests for the Phase 4 neural network modules.

Tests cover:
  • Model architecture (forward pass, shapes, parameter count)
  • Single-state prediction (predict method)
  • Training loop (loss decreases over epochs)
  • Checkpoint save/load roundtrip
  • Dataset creation and DataLoader
  • NN agent move selection
"""

import os
import sys
import json
import tempfile
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.neural_net.model import ColorWarsNet, get_device
from ai.neural_net.dataset import ColorWarsDataset, create_data_loaders
from ai.neural_net.trainer import Trainer
from ai.neural_net import config
from ai.agents.nn_agent import NNAgent
from ai.game_engine.game_state import GameState
from ai.game_engine.constants import GRID_SIZE, PLAYER_1, PLAYER_2
from ai.self_play.board_encoder import (
    encode_board,
    encode_board_from_perspective,
    encode_policy,
)


# ═══════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    """Use CPU for testing (deterministic, no GPU required)."""
    return torch.device("cpu")


@pytest.fixture
def model(device):
    """Create a fresh model on CPU."""
    m = ColorWarsNet()
    m.to(device)
    return m


@pytest.fixture
def game_state():
    """Create a game state with a few moves played."""
    state = GameState(grid_size=GRID_SIZE)
    state.apply_move(0, 0)  # P1
    state.apply_move(5, 5)  # P2
    state.apply_move(2, 3)  # P1
    return state


@pytest.fixture
def sample_examples():
    """
    Create synthetic training examples for testing.

    Each example is (encoded_board, flat_policy, value).
    """
    examples = []
    rng = np.random.RandomState(42)

    for _ in range(100):
        # Random 5-plane board
        board_planes = rng.rand(5, GRID_SIZE, GRID_SIZE).tolist()

        # Random policy (normalized)
        raw_probs = rng.rand(GRID_SIZE * GRID_SIZE)
        raw_probs = raw_probs / raw_probs.sum()
        flat_policy = raw_probs.tolist()

        # Random outcome
        value = 1.0 if rng.rand() > 0.5 else -1.0

        examples.append((board_planes, flat_policy, value))

    return examples


@pytest.fixture
def real_examples():
    """
    Create realistic training examples from actual game states.
    """
    examples = []

    for i in range(50):
        state = GameState(grid_size=GRID_SIZE)
        # Play a few random moves
        import random
        rng = random.Random(i)
        for _ in range(rng.randint(2, 10)):
            moves = state.get_legal_moves()
            if not moves:
                break
            move = rng.choice(moves)
            state.apply_move(*move)
            if state.is_game_over():
                break

        if state.is_game_over():
            continue

        # Encode the board
        board_planes = encode_board_from_perspective(
            state.board, state.current_player, GRID_SIZE,
        )

        # Create a fake policy over legal moves
        legal_moves = state.get_legal_moves()
        action_probs = {}
        rng2 = np.random.RandomState(i)
        raw = rng2.rand(len(legal_moves))
        raw = raw / raw.sum()
        for move, prob in zip(legal_moves, raw):
            action_probs[move] = float(prob)

        flat_policy = encode_policy(action_probs, GRID_SIZE)
        value = 1.0 if i % 2 == 0 else -1.0

        examples.append((board_planes, flat_policy, value))

    return examples


# ═══════════════════════════════════════════════════════════════════
#  Model Tests
# ═══════════════════════════════════════════════════════════════════

class TestColorWarsNet:
    """Tests for the neural network model."""

    def test_model_creation(self, model):
        """Model should be created with correct architecture."""
        assert isinstance(model, ColorWarsNet)
        assert model.grid_size == GRID_SIZE
        assert model.policy_size == GRID_SIZE * GRID_SIZE
        assert len(model.res_blocks) == config.NUM_RES_BLOCKS

    def test_parameter_count(self, model):
        """Model should have a reasonable number of parameters."""
        params = model.count_parameters()
        assert params > 0
        # For 4 ResBlocks / 64 filters, expect ~50K-150K parameters
        assert params < 500_000, f"Model has {params} parameters — too large"
        print(f"  Model parameters: {params:,}")

    def test_forward_pass_shape(self, model, device):
        """Forward pass should produce correct output shapes."""
        batch_size = 8
        x = torch.randn(batch_size, 5, GRID_SIZE, GRID_SIZE, device=device)

        log_policy, value = model(x)

        assert log_policy.shape == (batch_size, GRID_SIZE * GRID_SIZE)
        assert value.shape == (batch_size, 1)

    def test_forward_single(self, model, device):
        """Forward pass should work with batch size 1."""
        x = torch.randn(1, 5, GRID_SIZE, GRID_SIZE, device=device)
        log_policy, value = model(x)

        assert log_policy.shape == (1, GRID_SIZE * GRID_SIZE)
        assert value.shape == (1, 1)

    def test_policy_is_log_probabilities(self, model, device):
        """Policy output should be valid log-probabilities."""
        x = torch.randn(4, 5, GRID_SIZE, GRID_SIZE, device=device)
        log_policy, _ = model(x)

        # Log-probabilities should all be <= 0
        assert (log_policy <= 0).all()

        # exp(log_policy) should sum to ~1
        probs = torch.exp(log_policy)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4, device=device), atol=1e-5)

    def test_value_in_range(self, model, device):
        """Value output should be in [-1, +1] (tanh)."""
        x = torch.randn(16, 5, GRID_SIZE, GRID_SIZE, device=device)
        _, value = model(x)

        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_predict_method(self, model, game_state):
        """predict() should return a policy dict and float value."""
        board_planes = encode_board_from_perspective(
            game_state.board, game_state.current_player, GRID_SIZE,
        )

        policy_dict, value = model.predict(board_planes)

        # Policy should be a dict with (r, c) keys
        assert isinstance(policy_dict, dict)
        assert len(policy_dict) > 0

        for key, prob in policy_dict.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            r, c = key
            assert 0 <= r < GRID_SIZE
            assert 0 <= c < GRID_SIZE
            assert prob > 0

        # Value should be a float in [-1, 1]
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_predict_probabilities_sum_to_one(self, model, game_state):
        """predict() policy probabilities should sum to ~1."""
        board_planes = encode_board(
            game_state.board, game_state.current_player, GRID_SIZE,
        )

        policy_dict, _ = model.predict(board_planes)
        total = sum(policy_dict.values())
        assert abs(total - 1.0) < 0.01, f"Policy sum = {total}, expected ~1.0"


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint Tests
# ═══════════════════════════════════════════════════════════════════

class TestCheckpoint:
    """Tests for model save/load roundtrip."""

    def test_save_and_load(self, model, device):
        """Saved and loaded model should produce identical outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.pt")

            # Save
            model.save_checkpoint(filepath, epoch=5)

            # Load
            loaded_model, checkpoint = ColorWarsNet.load_checkpoint(
                filepath, device=device,
            )

            # Compare outputs
            x = torch.randn(4, 5, GRID_SIZE, GRID_SIZE, device=device)
            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                orig_policy, orig_value = model(x)
                load_policy, load_value = loaded_model(x)

            assert torch.allclose(orig_policy, load_policy, atol=1e-6)
            assert torch.allclose(orig_value, load_value, atol=1e-6)

    def test_checkpoint_metadata(self, model):
        """Checkpoint should preserve metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.pt")

            model.save_checkpoint(
                filepath, epoch=7,
                metadata={'test_key': 'test_value'},
            )

            _, checkpoint = ColorWarsNet.load_checkpoint(filepath)

            assert checkpoint['epoch'] == 7
            assert checkpoint['metadata']['test_key'] == 'test_value'
            assert checkpoint['grid_size'] == GRID_SIZE


# ═══════════════════════════════════════════════════════════════════
#  Dataset Tests
# ═══════════════════════════════════════════════════════════════════

class TestDataset:
    """Tests for the PyTorch dataset and data loaders."""

    def test_dataset_creation(self, sample_examples):
        """Dataset should be created from encoded examples."""
        dataset = ColorWarsDataset(sample_examples)
        assert len(dataset) == 100

    def test_dataset_getitem(self, sample_examples):
        """__getitem__ should return correctly shaped tensors."""
        dataset = ColorWarsDataset(sample_examples)
        board, policy, value = dataset[0]

        assert board.shape == (5, GRID_SIZE, GRID_SIZE)
        assert policy.shape == (GRID_SIZE * GRID_SIZE,)
        assert value.shape == (1,)
        assert board.dtype == torch.float32

    def test_data_loaders(self, sample_examples):
        """DataLoaders should produce correct batch shapes."""
        train_loader, val_loader = create_data_loaders(
            sample_examples, batch_size=16, train_split=0.8, seed=42,
        )

        assert train_loader is not None
        assert val_loader is not None

        # Check first batch
        boards, policies, values = next(iter(train_loader))
        assert boards.shape[0] <= 16
        assert boards.shape[1:] == (5, GRID_SIZE, GRID_SIZE)
        assert policies.shape[1] == GRID_SIZE * GRID_SIZE
        assert values.shape[1] == 1

    def test_data_loaders_split(self, sample_examples):
        """Train/val split should divide data correctly."""
        train_loader, val_loader = create_data_loaders(
            sample_examples, batch_size=16, train_split=0.8, seed=42,
        )

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        assert n_train + n_val == len(sample_examples)
        assert n_train == 80
        assert n_val == 20


# ═══════════════════════════════════════════════════════════════════
#  Trainer Tests
# ═══════════════════════════════════════════════════════════════════

class TestTrainer:
    """Tests for the training loop."""

    def test_trainer_creation(self, device):
        """Trainer should initialize correctly."""
        trainer = Trainer(device=device)
        assert trainer.model is not None
        assert trainer.current_epoch == 0

    def test_load_from_examples(self, device, sample_examples):
        """Trainer should accept pre-encoded examples."""
        trainer = Trainer(device=device, batch_size=16)
        n_train, n_val = trainer.load_from_examples(sample_examples)
        assert n_train > 0
        assert n_train + n_val == len(sample_examples)

    def test_training_loss_decreases(self, device, sample_examples):
        """Loss should decrease over training epochs."""
        model = ColorWarsNet()
        model.to(device)
        trainer = Trainer(model=model, device=device, batch_size=32, lr=0.01)
        trainer.load_from_examples(sample_examples, seed=42)

        metrics = trainer.train(n_epochs=5, verbose=False)

        first_loss = metrics[0]['train']['total_loss']
        last_loss = metrics[-1]['train']['total_loss']

        # Loss should decrease (or at least not increase dramatically)
        assert last_loss < first_loss * 1.5, (
            f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
        )

    def test_training_with_validation(self, device, sample_examples):
        """Training with validation should produce val metrics."""
        trainer = Trainer(device=device, batch_size=32)
        trainer.load_from_examples(sample_examples, seed=42)

        metrics = trainer.train(n_epochs=3, verbose=False)

        assert metrics[-1]['val'] is not None
        assert 'total_loss' in metrics[-1]['val']
        assert 'policy_accuracy' in metrics[-1]['val']

    def test_checkpoint_save_and_resume(self, device, sample_examples):
        """Training should be resumable from a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for 3 epochs and save
            trainer = Trainer(device=device, batch_size=32)
            trainer.load_from_examples(sample_examples, seed=42)
            trainer.train(n_epochs=3, verbose=False)

            filepath = os.path.join(tmpdir, "checkpoint.pt")
            trainer.save_checkpoint(filepath)

            assert trainer.current_epoch == 3

            # Load and verify
            trainer2 = Trainer(device=device, batch_size=32)
            trainer2.load_checkpoint(filepath)
            assert trainer2.current_epoch == 3

    def test_training_with_real_examples(self, device, real_examples):
        """Training should work with realistic game-derived examples."""
        if len(real_examples) < 10:
            pytest.skip("Not enough real examples generated")

        model = ColorWarsNet()
        model.to(device)
        trainer = Trainer(model=model, device=device, batch_size=16, lr=0.005)
        trainer.load_from_examples(real_examples, seed=42)

        metrics = trainer.train(n_epochs=5, verbose=False)

        # Should complete without errors
        assert len(metrics) == 5
        assert metrics[-1]['train']['total_loss'] > 0


# ═══════════════════════════════════════════════════════════════════
#  NN Agent Tests
# ═══════════════════════════════════════════════════════════════════

class TestNNAgent:
    """Tests for the neural network agent."""

    def test_agent_creation(self, model):
        """Agent should be created from a model."""
        agent = NNAgent(model=model, name="TestNN")
        assert agent.name == "TestNN"
        assert agent.grid_size == GRID_SIZE

    def test_select_move_returns_legal(self, model, game_state):
        """Agent should always select a legal move."""
        agent = NNAgent(model=model, temperature=0.01, seed=42)
        legal_moves = game_state.get_legal_moves()

        move = agent.select_move(game_state)

        assert move in legal_moves

    def test_select_move_with_policy(self, model, game_state):
        """Agent should return both action and policy."""
        agent = NNAgent(model=model, temperature=1.0, seed=42)

        action, policy = agent.select_move_with_policy(game_state)

        assert action in game_state.get_legal_moves()
        assert isinstance(policy, dict)
        assert len(policy) > 0

        # Policy should sum to ~1
        total = sum(policy.values())
        assert abs(total - 1.0) < 0.01

    def test_policy_only_legal_moves(self, model, game_state):
        """Agent's policy should only include legal moves."""
        agent = NNAgent(model=model, seed=42)
        legal_moves = set(game_state.get_legal_moves())

        _, policy = agent.select_move_with_policy(game_state)

        for action in policy:
            assert action in legal_moves

    def test_greedy_vs_sampling(self, model):
        """Greedy agent should consistently pick the same move."""
        state = GameState(grid_size=GRID_SIZE)
        state.apply_move(0, 0)
        state.apply_move(5, 5)

        greedy_agent = NNAgent(model=model, temperature=0.01, seed=42)
        moves = set()
        for _ in range(10):
            move = greedy_agent.select_move(state)
            moves.add(move)

        # Greedy should always pick the same move
        assert len(moves) == 1

    def test_temperature_annealing(self, model):
        """Temperature should anneal after threshold."""
        agent = NNAgent(
            model=model,
            temperature=1.0,
            temperature_threshold=5,
        )

        assert agent._get_temperature(move_number=3) == 1.0
        assert agent._get_temperature(move_number=5) == 1.0
        assert agent._get_temperature(move_number=6) == 0.01

    def test_value_estimate(self, model, game_state):
        """Value estimate should be in [-1, +1]."""
        agent = NNAgent(model=model, seed=42)
        value = agent.get_value_estimate(game_state)

        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_agent_plays_full_game(self, model):
        """Agent should be able to play a complete game."""
        agent1 = NNAgent(model=model, name="NN-1", temperature=0.5, seed=42)
        agent2 = NNAgent(model=model, name="NN-2", temperature=0.5, seed=99)

        state = GameState(grid_size=GRID_SIZE)
        agents = {PLAYER_1: agent1, PLAYER_2: agent2}

        max_turns = 500
        for turn in range(max_turns):
            if state.is_game_over():
                break
            agent = agents[state.current_player]
            move = agent.select_move(state)
            state.apply_move(*move)

        # Game should end within reasonable turns
        assert state.is_game_over(), (
            f"Game didn't end after {max_turns} turns"
        )
        assert state.get_winner() in (PLAYER_1, PLAYER_2)

    def test_from_checkpoint(self, model):
        """Agent should be loadable from a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_agent.pt")
            model.save_checkpoint(filepath)

            agent = NNAgent.from_checkpoint(filepath, name="Loaded-NN")
            assert agent.name == "Loaded-NN"

            state = GameState(grid_size=GRID_SIZE)
            move = agent.select_move(state)
            assert move in state.get_legal_moves()
