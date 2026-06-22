"""
game_state.py — Headless Color Wars game simulator.

This is the core game engine for AlphaZero training. It contains NO UI code
— no HTML, Canvas, DOM, or rendering. Everything runs in pure Python so we
can simulate thousands of games per second.

Architecture Overview:
─────────────────────
GameState represents a single snapshot of the game. It tracks:
  • board:          2D grid where each cell is [owner, dots]
  • current_player: whose turn it is (PLAYER_1 or PLAYER_2)
  • scores:         how many cells each player owns (maintained incrementally)
  • turn_count:     total moves made so far
  • game_over:      cached flag for terminal state
  • winner:         which player won (or None)

Key Design Decisions:
  1. Board is stored as list[list[list[int]]] — each cell is [owner, dots].
     We use mutable lists (not tuples) so chain reactions can modify in place.
  2. Scores are tracked incrementally during apply_move() instead of being
     recomputed by scanning the whole board. This matches the original JS logic.
  3. Chain reactions use BFS (breadth-first search) with deduplication,
     exactly matching the original JavaScript implementation.
  4. No numpy dependency — plain Python for simplicity. Numpy will be
     introduced in Phase 4 when we need tensor conversion for the neural net.

Complexity Analysis:
  • get_legal_moves():  O(n²) where n = grid_size
  • apply_move():       O(n² × k) worst case, where k = chain reaction rounds
  • is_game_over():     O(1) — uses cached scores
  • clone():            O(n²) — deep copies the board
"""

from collections import deque
from ai.game_engine.constants import (
    GRID_SIZE,
    EXPLOSION_THRESHOLD,
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    DIRECTIONS,
    MAX_CHAIN_ITERATIONS,
)


class GameState:
    """
    Complete state of a Color Wars game.

    This class is designed for AI training — it's fast, cloneable, and
    has no side effects. Every method is deterministic.

    Usage:
        state = GameState()
        moves = state.get_legal_moves()
        state.apply_move(moves[0][0], moves[0][1])
        if state.is_game_over():
            print(f"Winner: Player {state.get_winner()}")
    """

    def __init__(self, grid_size=GRID_SIZE):
        """
        Initialize an empty Color Wars board.

        Args:
            grid_size: Size of the square grid (default: 6 for a 6×6 board).

        The board is a 2D list of [owner, dots] pairs:
            board[row][col] = [owner, dots]
            - owner: EMPTY (0), PLAYER_1 (1), or PLAYER_2 (2)
            - dots:  number of dots in this cell (0 to EXPLOSION_THRESHOLD-1
                     in steady state, can temporarily exceed during chain reactions)

        Time complexity:  O(n²)
        Space complexity: O(n²)
        """
        self.grid_size = grid_size
        self.current_player = PLAYER_1  # Player 1 always goes first
        self.turn_count = 0             # Total moves made
        self.game_over = False          # Cached terminal state flag
        self.winner = None              # Winning player (None if game in progress)

        # Scores track how many cells each player owns.
        # Updated incrementally during apply_move() — never recomputed.
        self.scores = {PLAYER_1: 0, PLAYER_2: 0}

        # Build the empty board: each cell is [owner, dots]
        self.board = []
        for row in range(grid_size):
            row_list = []
            for col in range(grid_size):
                row_list.append([EMPTY, 0])  # [owner, dots]
            self.board.append(row_list)

    # ─── Legal Moves ─────────────────────────────────────────────────

    def get_legal_moves(self):
        """
        Return all valid moves for the current player.

        A move is valid if the cell is:
          • Empty (no owner), OR
          • Owned by the current player

        Returns:
            List of (row, col) tuples representing valid moves.

        Time complexity:  O(n²) — scans every cell
        Space complexity: O(n²) — worst case all cells are valid

        Example:
            >>> state = GameState()
            >>> moves = state.get_legal_moves()
            >>> len(moves)  # All 36 cells are empty, all are valid
            36
        """
        # If the game is over, no moves are available
        if self.game_over:
            return []

        moves = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                owner = self.board[row][col][0]
                # A cell is playable if it's empty or belongs to current player
                if owner == EMPTY or owner == self.current_player:
                    moves.append((row, col))
        return moves

    # ─── Apply Move ──────────────────────────────────────────────────

    def apply_move(self, row, col):
        """
        Apply a move: place a dot and resolve all chain reactions.

        This is the main game logic method. It:
          1. Validates the move
          2. Places a dot on the cell
          3. Resolves chain reactions (BFS)
          4. Checks win condition
          5. Switches to the other player

        Args:
            row: Row index (0-based)
            col: Column index (0-based)

        Raises:
            ValueError: If the move is invalid (wrong cell, game over, out of bounds)

        Time complexity:  O(n² × k) where k = number of chain reaction rounds
        Space complexity: O(n²) for the BFS queue in worst case

        The chain reaction logic exactly mirrors the JavaScript in index.html:
          • When a cell reaches EXPLOSION_THRESHOLD (4) dots:
            - It loses 4 dots
            - If dots reach 0, the cell becomes unowned
            - Each valid neighbor gains 1 dot and is captured by current player
            - Neighbors that reach threshold are added to the next explosion round
          • This repeats (BFS rounds) until no more explosions occur
        """
        # ── Validation ──
        if self.game_over:
            raise ValueError("Game is already over. Cannot apply move.")

        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            raise ValueError(
                f"Move ({row}, {col}) is out of bounds for a "
                f"{self.grid_size}×{self.grid_size} grid."
            )

        cell = self.board[row][col]
        if cell[0] != EMPTY and cell[0] != self.current_player:
            raise ValueError(
                f"Cell ({row}, {col}) is owned by Player {cell[0]}. "
                f"Player {self.current_player} cannot place here."
            )

        # ── Place the dot ──

        # If the cell was empty, the current player now claims it
        if cell[0] == EMPTY:
            cell[0] = self.current_player
            self.scores[self.current_player] += 1

        # Add one dot
        cell[1] += 1
        self.turn_count += 1

        # ── Resolve Chain Reactions (BFS) ──
        self._resolve_explosions(row, col)

        # ── Check Win Condition ──
        self._check_win_condition()

        # ── Switch Player ──
        if not self.game_over:
            self.current_player = (
                PLAYER_2 if self.current_player == PLAYER_1 else PLAYER_1
            )

    # ─── Chain Reaction Logic (BFS) ──────────────────────────────────

    def _resolve_explosions(self, start_row, start_col):
        """
        Resolve all chain reactions starting from a cell, using BFS.

        Algorithm (Breadth-First Explosion):
        ────────────────────────────────────
        1. Start with the initial cell in the explosion queue.
        2. For each round:
           a. Deduplicate cells in the queue (same cell might be queued
              multiple times from different neighbors).
           b. For each cell with dots >= EXPLOSION_THRESHOLD:
              - Remove EXPLOSION_THRESHOLD dots
              - If dots reach 0, cell becomes unowned (score decremented)
              - Each in-bounds neighbor:
                • Gets captured by current player (score adjustments)
                • Gains 1 dot
                • If it reaches threshold, added to next round's queue
        3. Repeat until a round produces no explosions.

        This matches the JS implementation's while-loop + deduplication
        pattern exactly (see index.html lines 242-313).

        Args:
            start_row: Row of the cell that just received a dot
            start_col: Column of the cell that just received a dot

        Time complexity:  O(n² × k) — each cell can explode at most a
                          bounded number of times per chain reaction cascade
        Space complexity: O(n²) for the queue
        """
        # The explosion queue starts with just the cell that was played
        explosions = [(start_row, start_col)]
        iteration_count = 0

        while explosions:
            # Safety valve to prevent infinite loops
            iteration_count += 1
            if iteration_count > MAX_CHAIN_ITERATIONS:
                break

            next_explosions = []

            # ── Deduplicate ──
            # If cell (2,3) appears twice in the queue, process it only once.
            # The cell may have accumulated dots from multiple neighbors.
            unique = {}
            for (r, c) in explosions:
                unique[(r, c)] = True  # dict preserves insertion order in Python 3.7+

            # ── Process each cell ──
            for (r, c) in unique:
                cell = self.board[r][c]

                # A cell might need to explode multiple times if it received
                # enough dots (e.g., 8 dots → explode twice)
                while cell[1] >= EXPLOSION_THRESHOLD:
                    # Remove threshold dots from this cell
                    cell[1] -= EXPLOSION_THRESHOLD

                    # If cell has 0 dots left, it becomes unowned
                    if cell[1] == 0:
                        cell[0] = EMPTY
                        self.scores[self.current_player] -= 1

                    # ── Spread to neighbors ──
                    for (dr, dc) in DIRECTIONS:
                        nr, nc = r + dr, c + dc

                        # Check bounds
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            neighbor = self.board[nr][nc]

                            # Capture logic: take ownership of the neighbor
                            if neighbor[0] == EMPTY:
                                # Empty cell → current player claims it
                                neighbor[0] = self.current_player
                                self.scores[self.current_player] += 1
                            elif neighbor[0] != self.current_player:
                                # Enemy cell → captured (swap ownership)
                                self.scores[neighbor[0]] -= 1
                                self.scores[self.current_player] += 1
                                neighbor[0] = self.current_player

                            # Add a dot to the neighbor
                            neighbor[1] += 1

                            # If neighbor now reaches threshold, it will
                            # explode in the next round
                            if neighbor[1] >= EXPLOSION_THRESHOLD:
                                next_explosions.append((nr, nc))

            # Move to next round of explosions
            explosions = next_explosions

    # ─── Win Condition ───────────────────────────────────────────────

    def _check_win_condition(self):
        """
        Check if the game is over.

        Win condition (matches the JS in index.html lines 322-334):
          • After the first full round of play (turn_count > 1):
            - If Player 1 has 0 cells → Player 2 wins
            - If Player 2 has 0 cells → Player 1 wins

        The turn_count > 1 guard prevents a false win on the very first move
        (when P2 naturally has 0 cells because they haven't played yet).

        Time complexity: O(1) — just checks cached scores
        """
        if self.turn_count > 1:
            if self.scores[PLAYER_1] == 0:
                self.game_over = True
                self.winner = PLAYER_2
            elif self.scores[PLAYER_2] == 0:
                self.game_over = True
                self.winner = PLAYER_1

    # ─── Game Over & Winner ──────────────────────────────────────────

    def is_game_over(self):
        """
        Check if the game has ended.

        Returns:
            True if the game is over, False otherwise.

        Time complexity: O(1)
        """
        return self.game_over

    def get_winner(self):
        """
        Get the winning player.

        Returns:
            PLAYER_1 (1) or PLAYER_2 (2) if game is over, None otherwise.

        Time complexity: O(1)
        """
        return self.winner

    # ─── Clone ───────────────────────────────────────────────────────

    def clone(self):
        """
        Create an independent deep copy of this game state.

        This is critical for MCTS (Phase 3) where we need to simulate
        moves without modifying the real game state.

        Returns:
            A new GameState that is a perfect copy. Modifying the clone
            will NOT affect the original, and vice versa.

        Time complexity:  O(n²)
        Space complexity: O(n²)
        """
        new_state = GameState.__new__(GameState)
        new_state.grid_size = self.grid_size
        new_state.current_player = self.current_player
        new_state.turn_count = self.turn_count
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        new_state.scores = {PLAYER_1: self.scores[PLAYER_1],
                            PLAYER_2: self.scores[PLAYER_2]}

        # Deep copy the board: each cell is a new [owner, dots] list
        new_state.board = []
        for row in self.board:
            new_row = []
            for cell in row:
                new_row.append([cell[0], cell[1]])
            new_state.board.append(new_row)

        return new_state

    def clone_state(self):
        """
        Alias for clone() — used by the MCTS module.

        Returns:
            A new independent deep copy of this GameState.
        """
        return self.clone()

    # ─── Board Printing (Console Debug) ──────────────────────────────

    def print_board(self):
        """
        Print the board state to the console in a readable ASCII format.

        Format:
          • Each cell shows: owner_initial + dot_count
          • "." = empty, "A" = Player 1, "B" = Player 2
          • Example: "A3" means Player 1 owns this cell with 3 dots

        Example output for a 6×6 board:
            ┌────┬────┬────┬────┬────┬────┐
            │ .0 │ A1 │ .0 │ .0 │ .0 │ .0 │
            ├────┼────┼────┼────┼────┼────┤
            │ .0 │ .0 │ B2 │ .0 │ .0 │ .0 │
            ...

        Time complexity: O(n²)
        """
        # Map owner IDs to display characters
        owner_chars = {EMPTY: '.', PLAYER_1: 'A', PLAYER_2: 'B'}

        # Column width for each cell
        cell_width = 4

        # Top border
        top = '┌' + '┬'.join(['─' * cell_width] * self.grid_size) + '┐'
        mid = '├' + '┼'.join(['─' * cell_width] * self.grid_size) + '┤'
        bot = '└' + '┴'.join(['─' * cell_width] * self.grid_size) + '┘'

        print(f"\n  Turn: {self.turn_count} | "
              f"Current Player: {self.current_player} | "
              f"Scores: P1={self.scores[PLAYER_1]} P2={self.scores[PLAYER_2]}")
        print(top)

        for row_idx, row in enumerate(self.board):
            cells = []
            for cell in row:
                owner_char = owner_chars[cell[0]]
                dots = cell[1]
                cells.append(f" {owner_char}{dots} ")
            print('│' + '│'.join(cells) + '│')
            if row_idx < self.grid_size - 1:
                print(mid)

        print(bot)

        if self.game_over:
            print(f"  *** GAME OVER — Player {self.winner} wins! ***\n")

    # ─── String Representation ───────────────────────────────────────

    def __repr__(self):
        """Compact string representation for debugging."""
        status = "OVER" if self.game_over else "IN_PROGRESS"
        return (
            f"GameState(turn={self.turn_count}, player={self.current_player}, "
            f"scores={{P1:{self.scores[PLAYER_1]}, P2:{self.scores[PLAYER_2]}}}, "
            f"status={status})"
        )
