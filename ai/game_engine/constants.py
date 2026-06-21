"""
constants.py — Game configuration constants for Color Wars.

These constants define the core game parameters. They are separated from
the game logic so that we can easily tweak them (e.g., try a 5×5 board)
without modifying the GameState class.

Used by: GameState, MCTS, Neural Network (future phases)
"""

# ─── Board Dimensions ────────────────────────────────────────────────
GRID_SIZE = 6  # 6×6 board (36 cells total)

# ─── Explosion Threshold ─────────────────────────────────────────────
# A cell explodes when its dot count reaches this value.
# Fixed at 4 for all cells (corners, edges, and center alike).
EXPLOSION_THRESHOLD = 4

# ─── Player Identifiers ──────────────────────────────────────────────
EMPTY = 0      # No owner
PLAYER_1 = 1   # First player (moves first)
PLAYER_2 = 2   # Second player

# ─── Directions for Neighbor Lookup ──────────────────────────────────
# (row_delta, col_delta) — Up, Right, Down, Left
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# ─── Game Limits ─────────────────────────────────────────────────────
# Safety limit to prevent infinite chain reaction loops.
# In a 6×6 board with threshold 4, the theoretical max chain depth
# is bounded, but we add a safety cap just in case.
MAX_CHAIN_ITERATIONS = 1000
