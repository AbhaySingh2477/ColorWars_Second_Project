"""
board_encoder.py — Convert game states to tensor-ready representations.

This module handles the encoding and decoding of game states and policies
between their natural Python representations and the flat numerical formats
needed for neural network training (Phase 4).

Board Encoding (5 planes):
──────────────────────────
  The board is encoded as 5 planes, each of size grid_size × grid_size:

    Plane 0: P1 ownership   — 1 if cell is owned by P1, else 0
    Plane 1: P1 dot count    — normalized dots for P1 cells (dots/threshold)
    Plane 2: P2 ownership   — 1 if cell is owned by P2, else 0
    Plane 3: P2 dot count    — normalized dots for P2 cells (dots/threshold)
    Plane 4: Current player  — all 1s if P1 to move, all 0s if P2 to move

  This is the standard approach from AlphaZero. The current player plane
  tells the network whose turn it is without needing separate models.

Policy Encoding:
────────────────
  The MCTS policy dict {(row, col): probability} is flattened to a
  1D vector of length grid_size² where index = row * grid_size + col.

  This is the format expected by the neural network's policy head.

Data Augmentation:
──────────────────
  Since Color Wars uses a uniform explosion threshold (4 for all cells),
  the board has 8-fold symmetry (4 rotations × 2 reflections). We can
  augment each training example by applying these transformations, which
  multiplies the effective training data by 8×.

  The 8 transformations are:
    0: identity
    1: rotate 90°
    2: rotate 180°
    3: rotate 270°
    4: horizontal flip
    5: flip + rotate 90°
    6: flip + rotate 180°
    7: flip + rotate 270°

Design Decision: No numpy dependency.
──────────────────────────────────────
  We use pure Python lists to stay consistent with the rest of the
  Phase 3 codebase. Phase 4 will introduce numpy/torch when the neural
  network is added. The encoding functions are designed so that
  converting to numpy is trivial: `np.array(encode_board(...))`.
"""

from ai.game_engine.constants import EMPTY, PLAYER_1, PLAYER_2, EXPLOSION_THRESHOLD


# ─── Board Encoding ─────────────────────────────────────────────────

def encode_board(board, current_player, grid_size):
    """
    Encode a board state into 5 binary/float planes.

    This is what the neural network sees as input. Each plane is
    a grid_size × grid_size 2D list of floats.

    Args:
        board:          2D list where board[r][c] = [owner, dots].
        current_player: PLAYER_1 or PLAYER_2 (whose turn it is).
        grid_size:      Size of the board.

    Returns:
        List of 5 planes, each a grid_size × grid_size 2D list:
          [p1_ownership, p1_dots, p2_ownership, p2_dots, current_player_plane]

    Time: O(n²) where n = grid_size
    """
    # Initialize 5 empty planes
    p1_own = [[0.0] * grid_size for _ in range(grid_size)]
    p1_dots = [[0.0] * grid_size for _ in range(grid_size)]
    p2_own = [[0.0] * grid_size for _ in range(grid_size)]
    p2_dots = [[0.0] * grid_size for _ in range(grid_size)]

    # Current player indicator: all 1s if P1 to move, all 0s if P2
    cp_val = 1.0 if current_player == PLAYER_1 else 0.0
    cp_plane = [[cp_val] * grid_size for _ in range(grid_size)]

    # Fill ownership and dot planes
    for r in range(grid_size):
        for c in range(grid_size):
            owner = board[r][c][0]
            dots = board[r][c][1]
            normalized_dots = dots / EXPLOSION_THRESHOLD  # Scale to [0, 1)

            if owner == PLAYER_1:
                p1_own[r][c] = 1.0
                p1_dots[r][c] = normalized_dots
            elif owner == PLAYER_2:
                p2_own[r][c] = 1.0
                p2_dots[r][c] = normalized_dots

    return [p1_own, p1_dots, p2_own, p2_dots, cp_plane]


def encode_board_from_perspective(board, current_player, grid_size):
    """
    Encode a board state from the current player's perspective.

    Instead of always putting P1 in planes 0-1 and P2 in planes 2-3,
    this puts the CURRENT player in planes 0-1 and the OPPONENT in
    planes 2-3. The current_player plane (plane 4) is always all 1s.

    This is the "canonical" encoding — the neural network always sees
    the board from the perspective of the player who's about to move.

    Args:
        board:          2D list where board[r][c] = [owner, dots].
        current_player: PLAYER_1 or PLAYER_2 (whose turn it is).
        grid_size:      Size of the board.

    Returns:
        List of 5 planes, each a grid_size × grid_size 2D list:
          [me_ownership, me_dots, opp_ownership, opp_dots, ones_plane]
    """
    opponent = PLAYER_2 if current_player == PLAYER_1 else PLAYER_1

    me_own = [[0.0] * grid_size for _ in range(grid_size)]
    me_dots = [[0.0] * grid_size for _ in range(grid_size)]
    opp_own = [[0.0] * grid_size for _ in range(grid_size)]
    opp_dots = [[0.0] * grid_size for _ in range(grid_size)]
    ones = [[1.0] * grid_size for _ in range(grid_size)]

    for r in range(grid_size):
        for c in range(grid_size):
            owner = board[r][c][0]
            dots = board[r][c][1]
            normalized_dots = dots / EXPLOSION_THRESHOLD

            if owner == current_player:
                me_own[r][c] = 1.0
                me_dots[r][c] = normalized_dots
            elif owner == opponent:
                opp_own[r][c] = 1.0
                opp_dots[r][c] = normalized_dots

    return [me_own, me_dots, opp_own, opp_dots, ones]


# ─── Policy Encoding / Decoding ─────────────────────────────────────

def encode_policy(action_probs, grid_size):
    """
    Encode an action probability dict to a flat probability vector.

    Args:
        action_probs: Dict mapping (row, col) → probability.
                      May be None (returns uniform vector).
        grid_size:    Size of the board.

    Returns:
        List of length grid_size² where index = row * grid_size + col.
        Probabilities sum to approximately 1.0.
    """
    total_cells = grid_size * grid_size
    flat = [0.0] * total_cells

    if action_probs is None:
        # Uniform policy (no MCTS data)
        uniform = 1.0 / total_cells
        return [uniform] * total_cells

    for (r, c), prob in action_probs.items():
        idx = r * grid_size + c
        flat[idx] = prob

    return flat


def decode_policy(flat_probs, grid_size):
    """
    Decode a flat probability vector back to an action probability dict.

    Args:
        flat_probs: List of length grid_size² with probabilities.
        grid_size:  Size of the board.

    Returns:
        Dict mapping (row, col) → probability.
        Only non-zero entries are included.
    """
    policy = {}
    for idx, prob in enumerate(flat_probs):
        if prob > 0:
            r = idx // grid_size
            c = idx % grid_size
            policy[(r, c)] = prob
    return policy


# ─── Data Augmentation (8-fold symmetry) ─────────────────────────────

def _rotate_board_90(board, grid_size):
    """Rotate a 2D board 90° clockwise."""
    rotated = [[None] * grid_size for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            rotated[c][grid_size - 1 - r] = board[r][c]
    return rotated


def _flip_board_horizontal(board, grid_size):
    """Flip a 2D board horizontally (left-right)."""
    flipped = [[None] * grid_size for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            flipped[r][grid_size - 1 - c] = board[r][c]
    return flipped


def _rotate_plane_90(plane, grid_size):
    """Rotate a single plane (2D list of floats) 90° clockwise."""
    rotated = [[0.0] * grid_size for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            rotated[c][grid_size - 1 - r] = plane[r][c]
    return rotated


def _flip_plane_horizontal(plane, grid_size):
    """Flip a single plane horizontally (left-right)."""
    flipped = [[0.0] * grid_size for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            flipped[r][grid_size - 1 - c] = plane[r][c]
    return flipped


def _transform_action(action, transform_id, grid_size):
    """
    Apply a symmetry transformation to a (row, col) action.

    Args:
        action:       (row, col) tuple.
        transform_id: Integer 0-7 specifying which transformation.
        grid_size:    Size of the board.

    Returns:
        Transformed (row, col) tuple.
    """
    r, c = action
    n = grid_size - 1

    if transform_id == 0:    # Identity
        return (r, c)
    elif transform_id == 1:  # Rotate 90° CW
        return (c, n - r)
    elif transform_id == 2:  # Rotate 180°
        return (n - r, n - c)
    elif transform_id == 3:  # Rotate 270° CW
        return (n - c, r)
    elif transform_id == 4:  # Horizontal flip
        return (r, n - c)
    elif transform_id == 5:  # Flip + Rotate 90°
        return (n - c, n - r)
    elif transform_id == 6:  # Flip + Rotate 180°
        return (n - r, c)
    elif transform_id == 7:  # Flip + Rotate 270°
        return (c, r)
    else:
        raise ValueError(f"Invalid transform_id: {transform_id}")


def _transform_policy(action_probs, transform_id, grid_size):
    """
    Apply a symmetry transformation to a policy dictionary.

    Args:
        action_probs:  Dict mapping (row, col) → probability.
        transform_id:  Integer 0-7 specifying which transformation.
        grid_size:     Size of the board.

    Returns:
        Transformed policy dict mapping transformed (row, col) → probability.
    """
    if action_probs is None:
        return None

    transformed = {}
    for action, prob in action_probs.items():
        new_action = _transform_action(action, transform_id, grid_size)
        transformed[new_action] = prob
    return transformed


def _transform_board(board, transform_id, grid_size):
    """
    Apply a symmetry transformation to a board state.

    The board is a 2D list of [owner, dots] cells.

    Args:
        board:         2D list where board[r][c] = [owner, dots].
        transform_id:  Integer 0-7 specifying which transformation.
        grid_size:     Size of the board.

    Returns:
        Transformed board (new 2D list, deep copied).
    """
    if transform_id == 0:
        # Identity — just deep copy
        return [[cell[:] for cell in row] for row in board]

    # Apply the transformation by mapping coordinates
    result = [[[0, 0] for _ in range(grid_size)] for _ in range(grid_size)]

    for r in range(grid_size):
        for c in range(grid_size):
            new_r, new_c = _transform_action((r, c), transform_id, grid_size)
            result[new_r][new_c] = board[r][c][:]

    return result


def augment_training_example(board_state, action_probs, outcome, grid_size):
    """
    Generate all 8 symmetry-augmented versions of a training example.

    Given one training example (board, policy, outcome), produces 8
    equivalent examples by applying the 8 symmetry transformations
    of the square board.

    Args:
        board_state:   2D list where board[r][c] = [owner, dots].
        action_probs:  Dict mapping (row, col) → probability, or None.
        outcome:       +1.0 or -1.0 (game result, unchanged by transforms).
        grid_size:     Size of the board.

    Returns:
        List of 8 tuples: (transformed_board, transformed_policy, outcome).
    """
    augmented = []

    for t_id in range(8):
        new_board = _transform_board(board_state, t_id, grid_size)
        new_policy = _transform_policy(action_probs, t_id, grid_size)
        augmented.append((new_board, new_policy, outcome))

    return augmented
