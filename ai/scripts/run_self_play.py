"""
run_self_play.py — Script to run self-play games and generate training data.

This is the command-line entry point for generating training data.
It uses RandomAgent vs RandomAgent to produce initial data that
will later be used to bootstrap the neural network (Phase 4).

Usage:
    cd ColorWars
    source .venv/bin/activate

    # Run 100 self-play games and save to disk
    python -m ai.scripts.run_self_play --games 100

    # Run with verbose output (see every move)
    python -m ai.scripts.run_self_play --games 3 --verbose

    # Custom grid size and save directory
    python -m ai.scripts.run_self_play --games 500 --grid-size 4 --save-dir data/custom

    # Reproducible run
    python -m ai.scripts.run_self_play --games 100 --seed 42
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.agents.random_agent import RandomAgent
from ai.self_play.self_play_manager import SelfPlayManager
from ai.self_play.game_record import GameRecord


def main():
    parser = argparse.ArgumentParser(
        description="Run Color Wars self-play and generate training data."
    )
    parser.add_argument(
        '--games', type=int, default=100,
        help='Number of self-play games (default: 100)'
    )
    parser.add_argument(
        '--grid-size', type=int, default=6,
        help='Grid size (default: 6)'
    )
    parser.add_argument(
        '--save-dir', type=str, default='data/self_play',
        help='Directory to save game records (default: data/self_play)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save games to disk (stats only)'
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

    # Create agents
    agent1 = RandomAgent(name="Random-P1", seed=args.seed)
    agent2 = RandomAgent(
        name="Random-P2",
        seed=args.seed + 1000 if args.seed is not None else None
    )

    # Create self-play manager
    manager = SelfPlayManager(grid_size=args.grid_size)

    # Run self-play
    save_dir = None if args.no_save else args.save_dir
    records = manager.run_self_play(
        agent1=agent1,
        agent2=agent2,
        n_games=args.games,
        save_dir=save_dir,
        verbose=args.verbose,
    )

    # Print statistics
    stats = manager.get_statistics(records)
    manager.print_statistics(stats)

    # Show a sample training example from the first game
    if records:
        examples = records[0].get_training_examples()
        print(f"  Sample training data from Game #{records[0].game_id}:")
        print(f"  Total examples in this game: {len(examples)}")
        if examples:
            board, action, outcome = examples[0]
            print(f"  First example:")
            print(f"    Action:  {action}")
            print(f"    Outcome: {outcome:+.1f} "
                  f"({'WIN' if outcome > 0 else 'LOSS'})")
            print(f"    Board (first row): {board[0]}")
        print()

    # Verify saved files
    if save_dir and not args.no_save:
        saved_files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
        print(f"  📁 Saved {len(saved_files)} game files to: {save_dir}/")

        # Verify one can be loaded back
        if saved_files:
            test_file = os.path.join(save_dir, saved_files[0])
            loaded = GameRecord.load(test_file)
            print(f"  ✅ Verification: Loaded {test_file} successfully")
            print(f"     Winner: Player {loaded.winner}, "
                  f"Turns: {loaded.total_turns}, "
                  f"Moves recorded: {len(loaded.moves)}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
