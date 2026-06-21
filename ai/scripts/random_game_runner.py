"""
random_game_runner.py — Run thousands of random Color Wars games.

This script validates that the game engine can:
  • Run many games without crashes or infinite loops
  • Always produce a winner
  • Handle all possible board states

It also collects statistics about game outcomes, which helps us understand
the game's dynamics before building the AI.

Usage:
    cd ColorWars

    # Run 1000 games (default)
    python -m ai.scripts.random_game_runner

    # Run 5000 games
    python -m ai.scripts.random_game_runner --games 5000

    # Run with verbose output (prints board each turn)
    python -m ai.scripts.random_game_runner --games 5 --verbose

    # Use a specific grid size
    python -m ai.scripts.random_game_runner --grid-size 4
"""

import sys
import os
import random
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.game_engine.game_state import GameState
from ai.game_engine.constants import PLAYER_1, PLAYER_2


def play_random_game(grid_size=6, verbose=False):
    """
    Play a single game where both players choose moves randomly.

    Args:
        grid_size: Size of the game grid
        verbose:   If True, print the board after every move

    Returns:
        Tuple of (winner, turn_count):
          - winner: PLAYER_1 or PLAYER_2
          - turn_count: how many moves the game lasted
    """
    state = GameState(grid_size=grid_size)

    if verbose:
        print("=" * 50)
        print("NEW GAME")
        print("=" * 50)
        state.print_board()

    while not state.is_game_over():
        # Get all legal moves for the current player
        moves = state.get_legal_moves()

        # Safety check — if no moves available but game isn't over,
        # something is wrong with the engine
        if not moves:
            print("ERROR: No legal moves but game is not over!")
            print(f"  State: {state}")
            state.print_board()
            return None, state.turn_count

        # Pick a random legal move
        row, col = random.choice(moves)

        if verbose:
            print(f"\nPlayer {state.current_player} plays: ({row}, {col})")

        # Apply the move
        state.apply_move(row, col)

        if verbose:
            state.print_board()

    return state.get_winner(), state.turn_count


def main():
    """Run multiple random games and collect statistics."""
    parser = argparse.ArgumentParser(
        description="Run random Color Wars games to test the engine."
    )
    parser.add_argument(
        '--games', type=int, default=1000,
        help='Number of games to simulate (default: 1000)'
    )
    parser.add_argument(
        '--grid-size', type=int, default=6,
        help='Grid size (default: 6 for 6×6)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print board state after every move'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ─── Statistics Tracking ─────────────────────────────────────────
    stats = {
        'total_games': args.games,
        'p1_wins': 0,
        'p2_wins': 0,
        'errors': 0,
        'total_turns': 0,
        'min_turns': float('inf'),
        'max_turns': 0,
        'turn_counts': [],
    }

    print(f"\n{'═' * 55}")
    print(f"  Color Wars — Random Game Runner")
    print(f"  Grid: {args.grid_size}×{args.grid_size} | "
          f"Games: {args.games} | "
          f"Seed: {args.seed or 'random'}")
    print(f"{'═' * 55}\n")

    start_time = time.time()

    # ─── Run Games ───────────────────────────────────────────────────
    for game_num in range(1, args.games + 1):
        winner, turns = play_random_game(
            grid_size=args.grid_size,
            verbose=args.verbose
        )

        if winner is None:
            stats['errors'] += 1
            continue

        # Update statistics
        if winner == PLAYER_1:
            stats['p1_wins'] += 1
        else:
            stats['p2_wins'] += 1

        stats['total_turns'] += turns
        stats['min_turns'] = min(stats['min_turns'], turns)
        stats['max_turns'] = max(stats['max_turns'], turns)
        stats['turn_counts'].append(turns)

        # Progress indicator (every 10% or every 100 games)
        if not args.verbose and game_num % max(1, args.games // 10) == 0:
            pct = (game_num / args.games) * 100
            print(f"  Progress: {game_num}/{args.games} ({pct:.0f}%)")

    elapsed = time.time() - start_time

    # ─── Print Results ───────────────────────────────────────────────
    completed = stats['p1_wins'] + stats['p2_wins']
    avg_turns = stats['total_turns'] / completed if completed > 0 else 0

    print(f"\n{'─' * 55}")
    print(f"  RESULTS")
    print(f"{'─' * 55}")
    print(f"  Games completed:  {completed}/{stats['total_games']}")
    print(f"  Errors:           {stats['errors']}")
    print(f"  Time elapsed:     {elapsed:.2f}s")
    print(f"  Games per second: {completed / elapsed:.0f}")
    print(f"{'─' * 55}")
    print(f"  Player 1 wins:    {stats['p1_wins']} "
          f"({stats['p1_wins']/completed*100:.1f}%)" if completed > 0 else "")
    print(f"  Player 2 wins:    {stats['p2_wins']} "
          f"({stats['p2_wins']/completed*100:.1f}%)" if completed > 0 else "")
    print(f"{'─' * 55}")
    print(f"  Avg game length:  {avg_turns:.1f} turns")
    print(f"  Min game length:  {stats['min_turns']} turns")
    print(f"  Max game length:  {stats['max_turns']} turns")
    print(f"{'═' * 55}\n")

    # ─── Sanity Checks ───────────────────────────────────────────────
    if stats['errors'] > 0:
        print("⚠️  WARNING: Some games had errors. Check the engine!")
        return 1
    else:
        print("✅ All games completed successfully. Engine is solid!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
