"""
self_play_manager.py — Orchestrates AI vs AI self-play games.

This is the "factory" that produces training data. In AlphaZero:
  1. SelfPlayManager runs games between two agents
  2. Each game produces a GameRecord with board states + actions + winner
  3. GameRecords are saved to disk
  4. The neural network trains on this data (Phase 4)
  5. The improved agent plays more games → better data → better model

Architecture:
─────────────
  SelfPlayManager
    ├── play_game(agent1, agent2)     → Single game → GameRecord
    ├── run_self_play(n_games, ...)   → Multiple games → List[GameRecord]
    ├── save_games(records, dir)      → Persist to disk
    └── get_statistics(records)       → Win rates, avg length, etc.

Design Decisions:
  • Agents are injected (dependency injection) — the manager doesn't
    care if it's RandomAgent, MCTSAgent, or NeuralNetAgent.
  • Statistics are computed from GameRecords, not tracked live, so
    they can be recalculated from saved data.
  • Game IDs are sequential within a run, making it easy to identify
    and sort games.

Time Complexity:
  • play_game():    O(T × n²) where T = game turns, n = grid_size
  • run_self_play(): O(G × T × n²) where G = number of games
"""

import os
import time
from ai.game_engine.game_state import GameState
from ai.game_engine.constants import PLAYER_1, PLAYER_2, GRID_SIZE
from ai.self_play.game_record import GameRecord


class SelfPlayManager:
    """
    Manages self-play game generation between two AI agents.

    This class is the bridge between agents and training data.
    It runs games, records every move, and produces GameRecord
    objects that can be saved and later converted to training examples.

    Usage:
        from ai.agents.random_agent import RandomAgent

        manager = SelfPlayManager(grid_size=6)
        agent1 = RandomAgent(name="Random-P1")
        agent2 = RandomAgent(name="Random-P2")

        # Run 100 games
        records = manager.run_self_play(
            agent1=agent1,
            agent2=agent2,
            n_games=100,
            save_dir="data/self_play"
        )

        # Print statistics
        stats = manager.get_statistics(records)
        manager.print_statistics(stats)
    """

    def __init__(self, grid_size=GRID_SIZE):
        """
        Args:
            grid_size: Size of the game grid (default: 6).
        """
        self.grid_size = grid_size
        self._game_counter = 0  # Running ID for games

    # ─── Single Game ─────────────────────────────────────────────────

    def play_game(self, agent1, agent2, verbose=False):
        """
        Play a single game between two agents and record everything.

        Args:
            agent1: Agent playing as Player 1.
            agent2: Agent playing as Player 2.
            verbose: If True, print the board after each move.

        Returns:
            GameRecord containing the full game history.

        Time Complexity:  O(T × n²) where T = number of turns
        Space Complexity: O(T × n²) — storing board snapshots
        """
        # Create a fresh game state
        state = GameState(grid_size=self.grid_size)

        # Create the game record
        self._game_counter += 1
        record = GameRecord(
            grid_size=self.grid_size,
            p1_agent=agent1.name,
            p2_agent=agent2.name,
            timestamp=time.time(),
            game_id=self._game_counter,
        )

        # Map player IDs to agents
        agents = {
            PLAYER_1: agent1,
            PLAYER_2: agent2,
        }

        if verbose:
            print(f"\n{'═' * 50}")
            print(f"  Game #{record.game_id}: {agent1.name} vs {agent2.name}")
            print(f"{'═' * 50}")
            state.print_board()

        # ── Game Loop ──
        while not state.is_game_over():
            current_agent = agents[state.current_player]

            # Record the board state BEFORE the move
            # (This is what the neural network will see as input)
            current_player = state.current_player

            # Let the agent choose a move
            action = current_agent.select_move(state)

            # Record the move (deep copies the board internally)
            record.add_move(
                board_state=state.board,
                action=action,
                player=current_player,
            )

            # Apply the move to advance the game
            state.apply_move(*action)

            if verbose:
                print(f"\n  {current_agent.name} (P{current_player}) "
                      f"plays: ({action[0]}, {action[1]})")
                state.print_board()

        # ── Game Over ──
        record.winner = state.get_winner()
        record.total_turns = state.turn_count

        if verbose:
            winner_name = agents[record.winner].name
            print(f"\n  🏆 Winner: {winner_name} "
                  f"(Player {record.winner}) in {record.total_turns} turns\n")

        return record

    # ─── Multiple Games ──────────────────────────────────────────────

    def run_self_play(self, agent1, agent2, n_games, save_dir=None,
                      verbose=False, progress_interval=None):
        """
        Run multiple self-play games and optionally save to disk.

        This is the main entry point for generating training data.

        Args:
            agent1:            Agent for Player 1.
            agent2:            Agent for Player 2.
            n_games:           Number of games to play.
            save_dir:          Directory to save game records (None = don't save).
            verbose:           Print board states during play.
            progress_interval: Print progress every N games (None = auto).

        Returns:
            List of GameRecord objects.

        Time Complexity:  O(G × T × n²)
        Space Complexity: O(G × T × n²) — all game records in memory
        """
        records = []

        if progress_interval is None:
            progress_interval = max(1, n_games // 10)

        print(f"\n{'═' * 55}")
        print(f"  Self-Play: {agent1.name} vs {agent2.name}")
        print(f"  Grid: {self.grid_size}×{self.grid_size} | Games: {n_games}")
        if save_dir:
            print(f"  Saving to: {save_dir}")
        print(f"{'═' * 55}\n")

        start_time = time.time()

        for game_num in range(1, n_games + 1):
            # Play one game
            record = self.play_game(agent1, agent2, verbose=verbose)
            records.append(record)

            # Save to disk if requested
            if save_dir:
                filepath = os.path.join(
                    save_dir,
                    f"game_{record.game_id:06d}.json"
                )
                record.save(filepath)

            # Progress reporting
            if game_num % progress_interval == 0 or game_num == n_games:
                elapsed = time.time() - start_time
                gps = game_num / elapsed if elapsed > 0 else 0
                print(f"  [{game_num}/{n_games}] "
                      f"{game_num/n_games*100:.0f}% | "
                      f"{gps:.0f} games/sec")

        elapsed = time.time() - start_time
        print(f"\n  Completed {n_games} games in {elapsed:.2f}s "
              f"({n_games/elapsed:.0f} games/sec)\n")

        return records

    # ─── Statistics ──────────────────────────────────────────────────

    @staticmethod
    def get_statistics(records):
        """
        Compute statistics from a list of game records.

        Args:
            records: List of GameRecord objects.

        Returns:
            Dictionary with statistics:
              - total_games:    Number of games played
              - p1_wins:        Player 1 win count
              - p2_wins:        Player 2 win count
              - p1_win_rate:    Player 1 win percentage
              - p2_win_rate:    Player 2 win percentage
              - avg_turns:      Average game length in turns
              - min_turns:      Shortest game
              - max_turns:      Longest game
              - total_examples: Total training examples generated
              - p1_agent:       Player 1 agent name
              - p2_agent:       Player 2 agent name

        Time Complexity: O(G) where G = number of game records
        """
        if not records:
            return {
                'total_games': 0,
                'p1_wins': 0, 'p2_wins': 0,
                'p1_win_rate': 0.0, 'p2_win_rate': 0.0,
                'avg_turns': 0.0, 'min_turns': 0, 'max_turns': 0,
                'total_examples': 0,
                'p1_agent': '', 'p2_agent': '',
            }

        total = len(records)
        p1_wins = sum(1 for r in records if r.winner == PLAYER_1)
        p2_wins = sum(1 for r in records if r.winner == PLAYER_2)
        turns = [r.total_turns for r in records]
        total_examples = sum(len(r.moves) for r in records)

        return {
            'total_games': total,
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
            'p1_win_rate': (p1_wins / total) * 100 if total > 0 else 0.0,
            'p2_win_rate': (p2_wins / total) * 100 if total > 0 else 0.0,
            'avg_turns': sum(turns) / len(turns) if turns else 0.0,
            'min_turns': min(turns) if turns else 0,
            'max_turns': max(turns) if turns else 0,
            'total_examples': total_examples,
            'p1_agent': records[0].p1_agent if records else '',
            'p2_agent': records[0].p2_agent if records else '',
        }

    @staticmethod
    def print_statistics(stats):
        """
        Pretty-print statistics to the console.

        Args:
            stats: Dictionary from get_statistics().
        """
        print(f"\n{'─' * 55}")
        print(f"  SELF-PLAY STATISTICS")
        print(f"{'─' * 55}")
        print(f"  Agents:           {stats['p1_agent']} vs {stats['p2_agent']}")
        print(f"  Total games:      {stats['total_games']}")
        print(f"{'─' * 55}")
        print(f"  P1 wins:          {stats['p1_wins']} "
              f"({stats['p1_win_rate']:.1f}%)")
        print(f"  P2 wins:          {stats['p2_wins']} "
              f"({stats['p2_win_rate']:.1f}%)")
        print(f"{'─' * 55}")
        print(f"  Avg game length:  {stats['avg_turns']:.1f} turns")
        print(f"  Min game length:  {stats['min_turns']} turns")
        print(f"  Max game length:  {stats['max_turns']} turns")
        print(f"{'─' * 55}")
        print(f"  Training examples: {stats['total_examples']}")
        print(f"  Avg examples/game: "
              f"{stats['total_examples']/stats['total_games']:.1f}"
              if stats['total_games'] > 0 else "  Avg examples/game: 0")
        print(f"{'═' * 55}\n")
