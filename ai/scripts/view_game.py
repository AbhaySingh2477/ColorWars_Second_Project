"""
view_game.py — Display a saved game record in human-readable format.

Reads a game JSON file and renders it as a sequence of ASCII boards
with move annotations, just like watching a replay.

Usage:
    cd ColorWars
    source .venv/bin/activate

    # View a game (shows summary + key moments)
    python -m ai.scripts.view_game data/self_play/game_000001.json

    # View every single move
    python -m ai.scripts.view_game data/self_play/game_000001.json --all-moves

    # View only first 10 moves
    python -m ai.scripts.view_game data/self_play/game_000001.json --moves 10

    # View specific move numbers
    python -m ai.scripts.view_game data/self_play/game_000001.json --move-range 50 60
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.self_play.game_record import GameRecord
from ai.game_engine.constants import EMPTY, PLAYER_1, PLAYER_2


def render_board(board_state, grid_size):
    """Render a board state as a readable ASCII grid."""
    owner_chars = {EMPTY: '.', PLAYER_1: 'A', PLAYER_2: 'B'}
    cell_width = 4

    top = '  ┌' + '┬'.join(['─' * cell_width] * grid_size) + '┐'
    mid = '  ├' + '┼'.join(['─' * cell_width] * grid_size) + '┤'
    bot = '  └' + '┴'.join(['─' * cell_width] * grid_size) + '┘'

    lines = [top]
    for row_idx, row in enumerate(board_state):
        cells = []
        for cell in row:
            owner_char = owner_chars.get(cell[0], '?')
            dots = cell[1]
            cells.append(f" {owner_char}{dots} ")
        lines.append('  │' + '│'.join(cells) + '│')
        if row_idx < grid_size - 1:
            lines.append(mid)
    lines.append(bot)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="View a saved Color Wars game in readable format."
    )
    parser.add_argument(
        'file', type=str,
        help='Path to the game JSON file'
    )
    parser.add_argument(
        '--all-moves', action='store_true',
        help='Show every single move (can be long!)'
    )
    parser.add_argument(
        '--moves', type=int, default=None,
        help='Show only the first N moves'
    )
    parser.add_argument(
        '--move-range', type=int, nargs=2, default=None,
        metavar=('START', 'END'),
        help='Show moves from START to END (1-indexed)'
    )

    args = parser.parse_args()

    # Load the game
    record = GameRecord.load(args.file)
    grid_size = record.grid_size

    # ── Game Summary ──
    print(f"\n{'═' * 55}")
    print(f"  GAME REPLAY — Game #{record.game_id}")
    print(f"{'═' * 55}")
    print(f"  File:       {args.file}")
    print(f"  Grid:       {grid_size}×{grid_size}")
    print(f"  Player 1:   {record.p1_agent} (shown as 'A')")
    print(f"  Player 2:   {record.p2_agent} (shown as 'B')")
    print(f"  Total turns: {record.total_turns}")
    print(f"  Winner:     Player {record.winner} "
          f"({'A' if record.winner == PLAYER_1 else 'B'}) 🏆")
    print(f"{'═' * 55}")

    # Determine which moves to show
    total = len(record.moves)

    if args.all_moves:
        start, end = 0, total
    elif args.moves:
        start, end = 0, min(args.moves, total)
    elif args.move_range:
        start = max(0, args.move_range[0] - 1)  # Convert to 0-indexed
        end = min(args.move_range[1], total)
    else:
        # Default: show first 5, last 5, and a few key moments
        show_indices = set()
        # First 5
        for i in range(min(5, total)):
            show_indices.add(i)
        # Last 5
        for i in range(max(0, total - 5), total):
            show_indices.add(i)
        # Every 25th move (milestones)
        for i in range(0, total, 25):
            show_indices.add(i)

        show_indices = sorted(show_indices)

        print(f"\n  Showing {len(show_indices)} key moves "
              f"(use --all-moves to see all {total}):\n")

        prev_idx = -1
        for idx in show_indices:
            move = record.moves[idx]
            player_char = 'A' if move.player == PLAYER_1 else 'B'

            if prev_idx >= 0 and idx > prev_idx + 1:
                print(f"  {'·' * 40}")
                print(f"  ... ({idx - prev_idx - 1} moves skipped) ...")
                print(f"  {'·' * 40}")

            print(f"\n  ── Move {idx + 1}/{total} ── "
                  f"Player {move.player} ({player_char}) → "
                  f"({move.action[0]}, {move.action[1]})")
            print(render_board(move.board_state, grid_size))

            prev_idx = idx

        # Show final result
        print(f"\n{'─' * 55}")
        winner_char = 'A' if record.winner == PLAYER_1 else 'B'
        print(f"  🏆 Player {record.winner} ({winner_char}) wins "
              f"in {record.total_turns} turns!")
        print(f"{'═' * 55}\n")
        return 0

    # Sequential display
    print(f"\n  Showing moves {start + 1} to {end} "
          f"(of {total} total):\n")

    for idx in range(start, end):
        move = record.moves[idx]
        player_char = 'A' if move.player == PLAYER_1 else 'B'

        print(f"\n  ── Move {idx + 1}/{total} ── "
              f"Player {move.player} ({player_char}) → "
              f"({move.action[0]}, {move.action[1]})")
        print(render_board(move.board_state, grid_size))

    # Final result
    print(f"\n{'─' * 55}")
    winner_char = 'A' if record.winner == PLAYER_1 else 'B'
    print(f"  🏆 Player {record.winner} ({winner_char}) wins "
          f"in {record.total_turns} turns!")
    print(f"{'═' * 55}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
