"""
game_record.py — Data structures for storing self-play game data.

In AlphaZero, the training pipeline works like this:
  1. Self-play generates games (this module stores the data)
  2. Training examples are extracted from those games
  3. The neural network trains on those examples

What We Store Per Game:
───────────────────────
  • GameRecord: Complete record of one game
    ├── moves:    List of (state_snapshot, action) at each turn
    ├── winner:   Which player won
    ├── metadata: Grid size, turn count, timestamps, agent names
    └── to training examples → (board_state, action, outcome)

What We Store Per Move (MoveRecord):
────────────────────────────────────
  • board_state: Snapshot of the board BEFORE the move was made
    - Stored as a flat list: [[owner, dots], [owner, dots], ...]
    - This is what the neural network will see as input
  • action:      The (row, col) move that was chosen
  • player:      Which player made this move

After the game ends, we know the winner, so we can label every move
with its outcome: +1 (this player won), -1 (this player lost).

File Format:
────────────
Games are saved as JSON files for easy inspection and loading.
Each file contains one GameRecord serialized to a dict.

Directory structure:
  data/self_play/
    ├── game_000001.json
    ├── game_000002.json
    └── ...
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


@dataclass
class MoveRecord:
    """
    Record of a single move in a game.

    Attributes:
        board_state: Board snapshot BEFORE this move.
                     Stored as list[list[int, int]] — flattened from
                     the 2D board. Each element is [owner, dots].
        action:      The (row, col) move that was made.
        player:      Which player made this move (1 or 2).
    """
    board_state: list       # 2D list: board[row][col] = [owner, dots]
    action: Tuple[int, int] # (row, col)
    player: int             # PLAYER_1 or PLAYER_2


@dataclass
class GameRecord:
    """
    Complete record of one self-play game.

    This is the primary data structure that flows through the pipeline:
      Self-Play → GameRecord → Training Examples → Neural Network

    Attributes:
        moves:       List of MoveRecord objects (one per turn).
        winner:      The winning player (1 or 2), or None for draws.
        grid_size:   Size of the board.
        total_turns: How many moves were played.
        p1_agent:    Name of Player 1's agent.
        p2_agent:    Name of Player 2's agent.
        timestamp:   When the game was played (Unix timestamp).
        game_id:     Unique identifier for this game.
    """
    moves: List[MoveRecord] = field(default_factory=list)
    winner: Optional[int] = None
    grid_size: int = 6
    total_turns: int = 0
    p1_agent: str = ""
    p2_agent: str = ""
    timestamp: float = 0.0
    game_id: int = 0

    def add_move(self, board_state, action, player):
        """
        Record a move.

        Args:
            board_state: 2D board list (will be deep-copied).
            action:      (row, col) tuple.
            player:      Player who made this move.
        """
        # Deep copy the board state so it doesn't get modified
        # by future moves
        snapshot = [[cell[:] for cell in row] for row in board_state]
        self.moves.append(MoveRecord(
            board_state=snapshot,
            action=action,
            player=player,
        ))

    def get_training_examples(self):
        """
        Convert this game record into training examples for the neural net.

        Each training example is a tuple:
            (board_state, action, outcome)

        Where:
          • board_state: The board BEFORE the move (2D list)
          • action:      The (row, col) move made
          • outcome:     +1 if this player won, -1 if this player lost

        This is the format AlphaZero uses:
          - The value head learns to predict outcome from board_state
          - The policy head learns to predict action from board_state

        Returns:
            List of (board_state, action, outcome) tuples.
        """
        examples = []
        for move_record in self.moves:
            # Determine outcome from this player's perspective
            if self.winner == move_record.player:
                outcome = 1.0   # This player won
            else:
                outcome = -1.0  # This player lost

            examples.append((
                move_record.board_state,
                move_record.action,
                outcome,
            ))
        return examples

    def to_dict(self):
        """
        Serialize to a JSON-compatible dictionary.

        We can't directly use dataclasses.asdict() because MoveRecord
        contains tuples which need special handling.
        """
        return {
            'game_id': self.game_id,
            'winner': self.winner,
            'grid_size': self.grid_size,
            'total_turns': self.total_turns,
            'p1_agent': self.p1_agent,
            'p2_agent': self.p2_agent,
            'timestamp': self.timestamp,
            'moves': [
                {
                    'board_state': mr.board_state,
                    'action': list(mr.action),
                    'player': mr.player,
                }
                for mr in self.moves
            ],
        }

    @classmethod
    def from_dict(cls, data):
        """
        Deserialize from a dictionary (loaded from JSON).

        Args:
            data: Dictionary with game record data.

        Returns:
            A new GameRecord object.
        """
        record = cls(
            winner=data['winner'],
            grid_size=data['grid_size'],
            total_turns=data['total_turns'],
            p1_agent=data.get('p1_agent', ''),
            p2_agent=data.get('p2_agent', ''),
            timestamp=data.get('timestamp', 0.0),
            game_id=data.get('game_id', 0),
        )
        for move_data in data['moves']:
            record.moves.append(MoveRecord(
                board_state=move_data['board_state'],
                action=tuple(move_data['action']),
                player=move_data['player'],
            ))
        return record

    def save(self, filepath):
        """
        Save this game record to a JSON file.

        Args:
            filepath: Path to save the JSON file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        """
        Load a game record from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            A GameRecord object.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
