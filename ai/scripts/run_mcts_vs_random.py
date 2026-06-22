"""
run_mcts_vs_random.py — Benchmark MCTS against RandomAgent.

This script pits the MCTSAgent against RandomAgent to demonstrate
that tree search is significantly stronger than random play.

Expected results with 200 simulations:
  • MCTS should win ~85-95% of games on a 6×6 board
  • On a 4×4 board, MCTS should win ~90-98%

Usage:
    cd ColorWars
    source .venv/bin/activate

    # Quick test (5 games, 4×4 board, 50 sims)
    python -m ai.scripts.run_mcts_vs_random --games 5 --grid-size 4 --sims 50

    # Full benchmark (10 games, 6×6 board, 200 sims)
    python -m ai.scripts.run_mcts_vs_random --games 10 --sims 200

    # Show MCTS thinking (verbose mode)
    python -m ai.scripts.run_mcts_vs_random --games 1 --verbose --sims 100
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai.game_engine.game_state import GameState
from ai.game_engine.constants import PLAYER_1, PLAYER_2
from ai.agents.mcts_agent import MCTSAgent
from ai.agents.random_agent import RandomAgent


def play_game(mcts_agent, random_agent, grid_size, verbose=False):
    """Play one game: MCTS (P1) vs Random (P2)."""
    state = GameState(grid_size=grid_size)
    agents = {PLAYER_1: mcts_agent, PLAYER_2: random_agent}
    move_count = 0

    while not state.is_game_over():
        agent = agents[state.current_player]
        move = agent.select_move(state)
        player = state.current_player

        state.apply_move(*move)
        move_count += 1

        if verbose and player == PLAYER_1:
            print(f"  MCTS plays ({move[0]},{move[1]})")
            mcts_agent.print_diagnostics()

    return state.get_winner(), move_count


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MCTS vs Random in Color Wars."
    )
    parser.add_argument('--games', type=int, default=5,
                        help='Number of games (default: 5)')
    parser.add_argument('--grid-size', type=int, default=4,
                        help='Grid size (default: 4 for speed)')
    parser.add_argument('--sims', type=int, default=100,
                        help='MCTS simulations per move (default: 100)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show MCTS diagnostics during play')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    print(f"\n{'═' * 55}")
    print(f"  MCTS vs RANDOM BENCHMARK")
    print(f"{'═' * 55}")
    print(f"  Grid:        {args.grid_size}×{args.grid_size}")
    print(f"  Games:       {args.games}")
    print(f"  MCTS sims:   {args.sims} per move")
    print(f"  Seed:        {args.seed}")
    print(f"{'═' * 55}\n")

    mcts_wins = 0
    random_wins = 0
    total_moves = 0
    start = time.time()

    for game_num in range(1, args.games + 1):
        mcts_agent = MCTSAgent(
            name="MCTS",
            n_simulations=args.sims,
            seed=args.seed + game_num,
        )
        random_agent = RandomAgent(
            name="Random",
            seed=args.seed + game_num + 1000,
        )

        game_start = time.time()
        winner, moves = play_game(
            mcts_agent, random_agent,
            args.grid_size, args.verbose,
        )
        game_time = time.time() - game_start

        total_moves += moves

        if winner == PLAYER_1:
            mcts_wins += 1
            result = "MCTS ✅"
        else:
            random_wins += 1
            result = "Random ⚡"

        print(f"  Game {game_num}/{args.games}: {result} "
              f"({moves} moves, {game_time:.1f}s)")

    elapsed = time.time() - start

    print(f"\n{'─' * 55}")
    print(f"  RESULTS")
    print(f"{'─' * 55}")
    print(f"  MCTS wins:    {mcts_wins}/{args.games} "
          f"({mcts_wins/args.games*100:.0f}%)")
    print(f"  Random wins:  {random_wins}/{args.games} "
          f"({random_wins/args.games*100:.0f}%)")
    print(f"  Avg moves:    {total_moves/args.games:.1f}")
    print(f"  Total time:   {elapsed:.1f}s")
    print(f"  Avg time/game: {elapsed/args.games:.1f}s")
    print(f"{'═' * 55}\n")


if __name__ == '__main__':
    main()
