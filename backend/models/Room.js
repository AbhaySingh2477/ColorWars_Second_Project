const mongoose = require('mongoose');

const roomSchema = new mongoose.Schema({
  code: {
    type: String,
    required: true,
    unique: true,
    uppercase: true,
    minlength: 6,
    maxlength: 6,
  },
  players: [{
    socketId: String,
    playerNumber: Number,
  }],
  board: {
    type: [[{
      owner: { type: Number, default: 0 },
      dots:  { type: Number, default: 0 },
    }]],
    default: undefined,
  },
  gridSize: {
    type: Number,
    default: 6,
  },
  currentPlayer: {
    type: Number,
    default: 1,
  },
  turnCount: {
    type: Number,
    default: 0,
  },
  scores: {
    type: Object,
    default: () => ({ 1: 0, 2: 0 }),
  },
  status: {
    type: String,
    enum: ['waiting', 'playing', 'finished'],
    default: 'waiting',
  },
  winner: {
    type: Number,
    default: null,
  },
  createdAt: {
    type: Date,
    default: Date.now,
    expires: 3600, // TTL: auto-delete after 1 hour
  },
});

/**
 * Initialize the board to a fresh 6×6 grid.
 */
roomSchema.methods.initBoard = function () {
  const size = this.gridSize;
  this.board = [];
  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) {
      row.push({ owner: 0, dots: 0 });
    }
    this.board.push(row);
  }
  this.currentPlayer = 1;
  this.turnCount = 0;
  this.scores = { 1: 0, 2: 0 };
  this.status = 'playing';
  this.winner = null;
};

/**
 * Apply a move and resolve all chain explosions.
 * Returns { valid, board, scores, currentPlayer, turnCount, winner, explosionSteps }
 */
roomSchema.methods.applyMove = function (row, col, playerNumber) {
  const size = this.gridSize;

  // Validate turn
  if (this.currentPlayer !== playerNumber) {
    return { valid: false, reason: 'Not your turn' };
  }
  if (this.status !== 'playing') {
    return { valid: false, reason: 'Game is not in progress' };
  }

  // Validate cell
  const cell = this.board[row][col];
  if (cell.owner !== 0 && cell.owner !== playerNumber) {
    return { valid: false, reason: 'Cell belongs to opponent' };
  }

  // Apply the dot
  if (cell.owner === 0) {
    cell.owner = playerNumber;
    this.scores[playerNumber] = (this.scores[playerNumber] || 0) + 1;
  }
  cell.dots++;
  this.turnCount++;

  // Resolve explosions (collect steps for animation)
  const explosionSteps = [];
  let explosions = [{ row, col }];
  const threshold = 4;

  while (explosions.length > 0) {
    let nextExplosions = [];
    const uniqueExplosions = [];
    const seen = new Set();
    for (const e of explosions) {
      const key = `${e.row},${e.col}`;
      if (!seen.has(key)) {
        seen.add(key);
        uniqueExplosions.push(e);
      }
    }

    let explodedCells = [];
    for (const { row: r, col: c } of uniqueExplosions) {
      const cellData = this.board[r][c];
      while (cellData.dots >= threshold) {
        explodedCells.push({ row: r, col: c });
        cellData.dots -= threshold;
        if (cellData.dots === 0) {
          cellData.owner = 0;
          this.scores[playerNumber]--;
        }

        const directions = [[-1, 0], [0, 1], [1, 0], [0, -1]];
        for (const [dx, dy] of directions) {
          const nr = r + dx;
          const nc = c + dy;
          if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
            const neighbor = this.board[nr][nc];
            if (neighbor.owner === 0) {
              neighbor.owner = playerNumber;
              this.scores[playerNumber]++;
            } else if (neighbor.owner !== playerNumber) {
              const otherPlayer = neighbor.owner;
              this.scores[otherPlayer]--;
              this.scores[playerNumber]++;
              neighbor.owner = playerNumber;
            }
            neighbor.dots++;
            if (neighbor.dots >= threshold) {
              nextExplosions.push({ row: nr, col: nc });
            }
          }
        }
      }
    }

    if (explodedCells.length > 0) {
      // Snapshot board state after this explosion step
      explosionSteps.push({
        explodedCells,
        boardSnapshot: this.board.map(r => r.map(c => ({ owner: c.owner, dots: c.dots }))),
      });
    }

    // Check win condition mid-explosion
    if (this.turnCount > 1) {
      if (this.scores[1] <= 0) {
        this.status = 'finished';
        this.winner = 2;
        break;
      }
      if (this.scores[2] <= 0) {
        this.status = 'finished';
        this.winner = 1;
        break;
      }
    }

    explosions = nextExplosions;
  }

  // Switch turn if game is still going
  if (this.status === 'playing') {
    this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
  }

  // Mark board as modified for Mongoose
  this.markModified('board');
  this.markModified('scores');

  return {
    valid: true,
    board: this.board,
    scores: this.scores,
    currentPlayer: this.currentPlayer,
    turnCount: this.turnCount,
    winner: this.winner,
    explosionSteps,
  };
};

module.exports = mongoose.model('Room', roomSchema);
