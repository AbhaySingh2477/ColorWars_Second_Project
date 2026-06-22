"""
run_mcts_self_play.py — Generate MCTS self-play training data.

This is the Phase 3 command-line entry point for generating high-quality
training data using MCTS vs MCTS self-play.

Each game produces training examples of the form:
    (board_state, mcts_policy, game_outcome)

These examples are used to train the neural network in Phase 4.

Usage:
    cd ColorWars
    source .venv/bin/activate

    # Quick test (3 games, small board, few sims)
    python -m ai.scripts.run_mcts_self_play --games 3 --sims 50 --grid-size 4

    # Standard training run (50 games, 6×6 board, 200 sims)
    python -m ai.scripts.run_mcts_self_play --games 50 --sims 200

    # Full training run with augmentation analysis
    python -m ai.scripts.run_mcts_self_play \\
        --games 100 --sims 200 \\
        --temp-threshold 8 \\
        --save-dir data/mcts_self_play \\
        --augment

    # Reproducible run
    python -m ai.scripts.run_mcts_self_play --games 50 --sims 100 --seed 42

    # Verbose mode (see every move + MCTS policy)
    python -m ai.scripts.run_mcts_self_play --games 1 --sims 50 --verbose
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.self_play.training_pipeline import TrainingPipeline
from ai.self_play.game_record import GameRecord


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCTS self-play training data for Color Wars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:   python -m ai.scripts.run_mcts_self_play --games 3 --sims 50 --grid-size 4
  Training run: python -m ai.scripts.run_mcts_self_play --games 50 --sims 200
  With augment: python -m ai.scripts.run_mcts_self_play --games 50 --sims 200 --augment
        """,
    )

    # ── Game Parameters ──
    parser.add_argument(
        '--games', type=int, default=10,
        help='Number of self-play games (default: 10)'
    )
    parser.add_argument(
        '--grid-size', type=int, default=6,
        help='Grid size (default: 6)'
    )

    # ── MCTS Parameters ──
    parser.add_argument(
        '--sims', type=int, default=100,
        help='MCTS simulations per move (default: 100)'
    )
    parser.add_argument(
        '--c-puct', type=float, default=1.41,
        help='UCB exploration constant (default: 1.41)'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Initial temperature for exploration (default: 1.0)'
    )
    parser.add_argument(
        '--temp-threshold', type=int, default=8,
        help='Move number at which temperature drops to greedy (default: 8)'
    )

    # ── Output ──
    parser.add_argument(
        '--save-dir', type=str, default='data/mcts_self_play',
        help='Directory to save game records (default: data/mcts_self_play)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save games to disk (stats only)'
    )

    # ── Data Processing ──
    parser.add_argument(
        '--augment', action='store_true',
        help='Show augmented training example count (8× via symmetry)'
    )
    parser.add_argument(
        '--no-swap', action='store_true',
        help='Disable color swapping between games'
    )

    # ── Misc ──
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print board state and MCTS policy for every move'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # ── Banner ──
    print(f"\n{'═' * 60}")
    print(f"  COLOR WARS — MCTS SELF-PLAY TRAINING DATA GENERATION")
    print(f"{'═' * 60}")
    print(f"  Grid:              {args.grid_size}×{args.grid_size}")
    print(f"  Games:             {args.games}")
    print(f"  MCTS sims/move:    {args.sims}")
    print(f"  c_puct:            {args.c_puct}")
    print(f"  Temperature:       {args.temperature} → greedy after move "
          f"{args.temp_threshold}")
    print(f"  Color swapping:    {'OFF' if args.no_swap else 'ON'}")
    print(f"  Data augmentation: {'ON (8× symmetry)' if args.augment else 'OFF'}")
    print(f"  Seed:              {args.seed or 'random'}")
    if not args.no_save:
        print(f"  Save directory:    {args.save_dir}")
    print(f"{'═' * 60}\n")

    # ── Run Pipeline ──
    pipeline = TrainingPipeline(grid_size=args.grid_size)

    save_dir = None if args.no_save else args.save_dir

    records = pipeline.generate(
        n_games=args.games,
        n_simulations=args.sims,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_threshold=args.temp_threshold,
        save_dir=save_dir,
        seed=args.seed,
        verbose=args.verbose,
        swap_colors=not args.no_swap,
    )

    # ── Extract Examples ──
    examples = pipeline.extract_examples(records, augment=False)
    stats = pipeline.get_data_stats(examples)
    pipeline.print_data_stats(stats)

    if args.augment:
        aug_examples = pipeline.extract_examples(records, augment=True)
        aug_stats = pipeline.get_data_stats(aug_examples)
        print(f"  With 8× augmentation:")
        print(f"    Raw examples:       {stats['total_examples']}")
        print(f"    Augmented examples: {aug_stats['total_examples']}")
        print(f"    Multiplier:         "
              f"{aug_stats['total_examples']/stats['total_examples']:.1f}×"
              if stats['total_examples'] > 0 else "    Multiplier: N/A")
        print()

    # ── Show Sample Example ──
    if records:
        examples = records[0].get_training_examples()
        print(f"  Sample training data from Game #{records[0].game_id}:")
        print(f"  Total examples in this game: {len(examples)}")
        if examples:
            board, policy, outcome = examples[0]
            print(f"  First example:")
            print(f"    Outcome: {outcome:+.1f} "
                  f"({'WIN' if outcome > 0 else 'LOSS'})")
            print(f"    Board (first row): {board[0]}")
            if policy:
                top_moves = sorted(policy.items(),
                                   key=lambda x: x[1], reverse=True)[:5]
                print(f"    Policy top-5:")
                for (r, c), prob in top_moves:
                    print(f"      ({r},{c}): {prob:.3f}")
            else:
                print(f"    Policy: None (no MCTS data)")
        print()

    # ── Verify Saved Files ──
    if save_dir and not args.no_save:
        saved_files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
        print(f"  📁 Saved {len(saved_files)} game files to: {save_dir}/")

        if saved_files:
            test_file = os.path.join(save_dir, sorted(saved_files)[-1])
            loaded = GameRecord.load(test_file)
            print(f"  ✅ Verification: Loaded {test_file}")
            print(f"     Winner: Player {loaded.winner}, "
                  f"Turns: {loaded.total_turns}, "
                  f"Policy data: {'Yes' if loaded.has_policy else 'No'}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
