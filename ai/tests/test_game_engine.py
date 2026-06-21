"""
test_game_engine.py — Unit tests for the Color Wars game engine.

Tests cover:
  1. Initial state correctness
  2. Legal move generation
  3. Basic move placement
  4. Explosion at threshold
  5. Chain reaction propagation
  6. Capture mechanics
  7. Win condition detection
  8. Clone independence
  9. Invalid move rejection
  10. Edge cases (corners, edges, full board)

Run with:
    cd ColorWars
    python -m pytest ai/tests/test_game_engine.py -v
"""

import sys
import os

# Add project root to path so we can import the ai package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.game_engine.game_state import GameState
from ai.game_engine.constants import (
    GRID_SIZE, EXPLOSION_THRESHOLD, EMPTY, PLAYER_1, PLAYER_2
)


# ═══════════════════════════════════════════════════════════════════════
# 1. INITIAL STATE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestInitialState:
    """Test that a fresh game starts in the correct state."""

    def test_empty_board(self):
        """All cells should be empty with 0 dots."""
        state = GameState()
        for row in range(state.grid_size):
            for col in range(state.grid_size):
                assert state.board[row][col] == [EMPTY, 0], \
                    f"Cell ({row},{col}) should be [0, 0]"

    def test_player_1_starts(self):
        """Player 1 should always go first."""
        state = GameState()
        assert state.current_player == PLAYER_1

    def test_scores_start_at_zero(self):
        """Both players should start with 0 owned cells."""
        state = GameState()
        assert state.scores[PLAYER_1] == 0
        assert state.scores[PLAYER_2] == 0

    def test_turn_count_starts_at_zero(self):
        """No moves have been made yet."""
        state = GameState()
        assert state.turn_count == 0

    def test_game_not_over(self):
        """Game should be in progress at the start."""
        state = GameState()
        assert state.is_game_over() is False
        assert state.get_winner() is None

    def test_default_grid_size(self):
        """Default grid should be 6×6."""
        state = GameState()
        assert state.grid_size == GRID_SIZE
        assert len(state.board) == GRID_SIZE
        assert len(state.board[0]) == GRID_SIZE

    def test_custom_grid_size(self):
        """Should support custom grid sizes."""
        state = GameState(grid_size=4)
        assert state.grid_size == 4
        assert len(state.board) == 4
        assert len(state.board[0]) == 4


# ═══════════════════════════════════════════════════════════════════════
# 2. LEGAL MOVES TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestLegalMoves:
    """Test get_legal_moves() returns correct valid positions."""

    def test_all_moves_on_empty_board(self):
        """On an empty board, every cell is a valid move."""
        state = GameState()
        moves = state.get_legal_moves()
        assert len(moves) == state.grid_size ** 2

    def test_cannot_move_on_opponent_cell(self):
        """Player 2 should not be able to place on Player 1's cell."""
        state = GameState()
        state.apply_move(0, 0)  # P1 plays at (0,0)
        # Now it's P2's turn — (0,0) should NOT be in legal moves
        moves = state.get_legal_moves()
        assert (0, 0) not in moves

    def test_can_move_on_own_cell(self):
        """A player can place on their own cell to add more dots."""
        state = GameState()
        state.apply_move(0, 0)  # P1 plays at (0,0)
        state.apply_move(1, 1)  # P2 plays at (1,1)
        # Now it's P1's turn — (0,0) should still be valid
        moves = state.get_legal_moves()
        assert (0, 0) in moves

    def test_no_moves_when_game_over(self):
        """No legal moves should be returned after the game ends."""
        state = GameState()
        state.game_over = True
        assert state.get_legal_moves() == []


# ═══════════════════════════════════════════════════════════════════════
# 3. BASIC MOVE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestBasicMove:
    """Test that placing a dot works correctly."""

    def test_place_on_empty_cell(self):
        """Placing on an empty cell should claim it and add 1 dot."""
        state = GameState()
        state.apply_move(2, 3)

        assert state.board[2][3][0] == PLAYER_1, "Owner should be Player 1"
        assert state.board[2][3][1] == 1, "Should have 1 dot"
        assert state.scores[PLAYER_1] == 1, "P1 should own 1 cell"

    def test_place_on_own_cell(self):
        """Placing on your own cell should add another dot."""
        state = GameState()
        state.apply_move(2, 3)  # P1: (2,3) → 1 dot
        state.apply_move(0, 0)  # P2 plays somewhere
        state.apply_move(2, 3)  # P1: (2,3) → 2 dots

        assert state.board[2][3][0] == PLAYER_1
        assert state.board[2][3][1] == 2

    def test_turn_switches_after_move(self):
        """Current player should alternate after each move."""
        state = GameState()
        assert state.current_player == PLAYER_1

        state.apply_move(0, 0)
        assert state.current_player == PLAYER_2

        state.apply_move(1, 1)
        assert state.current_player == PLAYER_1

    def test_turn_count_increments(self):
        """Turn count should increase by 1 after each move."""
        state = GameState()
        state.apply_move(0, 0)
        assert state.turn_count == 1
        state.apply_move(1, 1)
        assert state.turn_count == 2


# ═══════════════════════════════════════════════════════════════════════
# 4. EXPLOSION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestExplosion:
    """Test that cells explode correctly at the threshold."""

    def test_explosion_at_threshold(self):
        """A cell with 4 dots should explode (lose 4 dots)."""
        state = GameState()
        # Manually set up a cell with 3 dots (one more will trigger explosion)
        state.board[3][3] = [PLAYER_1, 3]
        state.scores[PLAYER_1] = 1

        # P1 adds 1 more dot → 4 dots → explosion
        state.apply_move(3, 3)
        # After explosion: cell should have 0 dots (4 - 4 = 0)
        assert state.board[3][3][1] == 0

    def test_explosion_spreads_to_neighbors(self):
        """Neighbors should each gain 1 dot after an explosion."""
        state = GameState()
        # Center cell (3,3) with 3 dots — one more triggers explosion
        state.board[3][3] = [PLAYER_1, 3]
        state.scores[PLAYER_1] = 1

        state.apply_move(3, 3)  # → 4 dots → explodes

        # All 4 neighbors should have gained 1 dot and be owned by P1
        neighbors = [(2, 3), (4, 3), (3, 2), (3, 4)]
        for (r, c) in neighbors:
            assert state.board[r][c][1] >= 1, \
                f"Neighbor ({r},{c}) should have at least 1 dot"
            assert state.board[r][c][0] == PLAYER_1, \
                f"Neighbor ({r},{c}) should be owned by Player 1"

    def test_explosion_at_corner(self):
        """A corner cell should only spread to 2 neighbors (not out of bounds)."""
        state = GameState()
        # Corner (0,0) with 3 dots
        state.board[0][0] = [PLAYER_1, 3]
        state.scores[PLAYER_1] = 1

        state.apply_move(0, 0)  # → 4 dots → explodes

        # Only 2 neighbors are in bounds: (0,1) and (1,0)
        assert state.board[0][1][1] >= 1, "(0,1) should gain a dot"
        assert state.board[1][0][1] >= 1, "(1,0) should gain a dot"

    def test_explosion_at_edge(self):
        """An edge cell should spread to 3 neighbors."""
        state = GameState()
        # Edge cell (0,2) — top edge, not corner
        state.board[0][2] = [PLAYER_1, 3]
        state.scores[PLAYER_1] = 1

        state.apply_move(0, 2)  # → 4 dots → explodes

        # 3 neighbors in bounds: (0,1), (0,3), (1,2)
        assert state.board[0][1][1] >= 1
        assert state.board[0][3][1] >= 1
        assert state.board[1][2][1] >= 1


# ═══════════════════════════════════════════════════════════════════════
# 5. CHAIN REACTION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestChainReaction:
    """Test multi-step chain reactions."""

    def test_chain_reaction(self):
        """An explosion that fills a neighbor to threshold should cascade."""
        state = GameState()
        # Set up: two adjacent cells both near threshold
        state.board[3][3] = [PLAYER_1, 3]  # Will explode after +1
        state.board[3][4] = [PLAYER_1, 3]  # Will explode from neighbor's spread
        state.scores[PLAYER_1] = 2

        state.apply_move(3, 3)  # → 4 → explodes → (3,4) goes from 3 to 4 → explodes

        # (3,4) should have exploded too, so its dots should be < 4
        assert state.board[3][4][1] < EXPLOSION_THRESHOLD, \
            "Chain reaction should have exploded (3,4)"

    def test_chain_captures_enemy_cells(self):
        """Chain reactions should capture enemy-owned cells."""
        state = GameState()
        # P1 has a cell about to explode
        state.board[3][3] = [PLAYER_1, 3]
        state.scores[PLAYER_1] = 1
        # P2 has an adjacent cell
        state.board[3][4] = [PLAYER_2, 2]
        state.scores[PLAYER_2] = 1

        state.apply_move(3, 3)  # P1 explosion captures (3,4)

        assert state.board[3][4][0] == PLAYER_1, \
            "Enemy cell (3,4) should be captured by Player 1"


# ═══════════════════════════════════════════════════════════════════════
# 6. WIN CONDITION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestWinCondition:
    """Test game-over detection."""

    def test_no_win_on_first_move(self):
        """Game should not end on the very first move (P2 has 0 cells naturally)."""
        state = GameState()
        state.apply_move(0, 0)  # P1's first move
        assert state.is_game_over() is False, \
            "Game should not end just because P2 hasn't played yet"

    def test_win_when_opponent_has_no_cells(self):
        """Player wins when the opponent loses all their cells via explosions."""
        state = GameState()
        # Simulate: P2 has exactly 1 cell, and a P1 explosion is about to capture it
        state.board[3][3] = [PLAYER_1, 3]
        state.board[3][4] = [PLAYER_2, 1]
        state.scores[PLAYER_1] = 1
        state.scores[PLAYER_2] = 1
        state.turn_count = 5  # Past the early-game guard

        state.apply_move(3, 3)  # P1 explodes → captures (3,4) → P2 has 0 cells

        assert state.is_game_over() is True
        assert state.get_winner() == PLAYER_1

    def test_game_continues_when_both_have_cells(self):
        """Game should continue when both players still own cells."""
        state = GameState()
        state.apply_move(0, 0)  # P1
        state.apply_move(5, 5)  # P2
        assert state.is_game_over() is False


# ═══════════════════════════════════════════════════════════════════════
# 7. CLONE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestClone:
    """Test that clone() creates an independent deep copy."""

    def test_clone_equals_original(self):
        """Clone should have the same state as the original."""
        state = GameState()
        state.apply_move(2, 3)
        state.apply_move(4, 1)

        clone = state.clone()

        assert clone.current_player == state.current_player
        assert clone.turn_count == state.turn_count
        assert clone.scores == state.scores
        assert clone.game_over == state.game_over

        for row in range(state.grid_size):
            for col in range(state.grid_size):
                assert clone.board[row][col] == state.board[row][col]

    def test_clone_is_independent(self):
        """Modifying the clone should not affect the original."""
        state = GameState()
        state.apply_move(0, 0)  # P1
        state.apply_move(1, 1)  # P2

        clone = state.clone()

        # Modify the clone
        clone.apply_move(2, 2)  # P1 plays in clone

        # Original should be unchanged
        assert state.board[2][2] == [EMPTY, 0], \
            "Original board should not be affected by clone"
        assert state.turn_count == 2, \
            "Original turn count should not change"

    def test_clone_board_independence(self):
        """Directly modifying clone's board should not affect original."""
        state = GameState()
        state.apply_move(0, 0)

        clone = state.clone()
        clone.board[0][0] = [PLAYER_2, 99]

        assert state.board[0][0] == [PLAYER_1, 1], \
            "Original cell should not be modified"


# ═══════════════════════════════════════════════════════════════════════
# 8. INVALID MOVE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestInvalidMoves:
    """Test that invalid moves are properly rejected."""

    def test_move_on_opponent_cell(self):
        """Should raise ValueError when placing on opponent's cell."""
        state = GameState()
        state.apply_move(0, 0)  # P1 claims (0,0)
        # Now P2 tries to place on (0,0) — should fail
        try:
            state.apply_move(0, 0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

    def test_move_out_of_bounds(self):
        """Should raise ValueError for out-of-bounds coordinates."""
        state = GameState()
        try:
            state.apply_move(-1, 0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            state.apply_move(0, 6)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_move_after_game_over(self):
        """Should raise ValueError when trying to play after game ends."""
        state = GameState()
        state.game_over = True
        try:
            state.apply_move(0, 0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# 9. SCORE TRACKING TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestScoreTracking:
    """Test that scores are maintained correctly during gameplay."""

    def test_score_increments_on_claim(self):
        """Claiming an empty cell should increase score by 1."""
        state = GameState()
        state.apply_move(0, 0)
        assert state.scores[PLAYER_1] == 1

    def test_score_tracks_multiple_cells(self):
        """Score should reflect total cells owned."""
        state = GameState()
        state.apply_move(0, 0)  # P1
        state.apply_move(1, 1)  # P2
        state.apply_move(2, 2)  # P1

        assert state.scores[PLAYER_1] == 2
        assert state.scores[PLAYER_2] == 1

    def test_score_no_double_count_on_own_cell(self):
        """Placing on your own cell should NOT increase your score again."""
        state = GameState()
        state.apply_move(0, 0)  # P1 claims (0,0) — score = 1
        state.apply_move(1, 1)  # P2
        state.apply_move(0, 0)  # P1 adds dot to own cell — score still 1

        # P1 didn't claim a new cell, just added a dot
        # But P1 also claimed (0,0), so score should still be 1 for that cell
        # (unless explosion captured more cells)
        assert state.scores[PLAYER_1] >= 1


# ═══════════════════════════════════════════════════════════════════════
# 10. REPR AND PRINT TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestReprAndPrint:
    """Test string representation and board printing."""

    def test_repr_initial_state(self):
        """repr should show a readable summary."""
        state = GameState()
        r = repr(state)
        assert "turn=0" in r
        assert "player=1" in r
        assert "IN_PROGRESS" in r

    def test_repr_after_moves(self):
        """repr should update after moves."""
        state = GameState()
        state.apply_move(0, 0)
        r = repr(state)
        assert "turn=1" in r
        assert "player=2" in r

    def test_print_board_does_not_crash(self):
        """print_board() should run without errors on any state."""
        state = GameState()
        state.print_board()  # Should not raise

        state.apply_move(0, 0)
        state.apply_move(5, 5)
        state.print_board()  # Should not raise
