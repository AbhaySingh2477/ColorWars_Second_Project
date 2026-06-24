require('dotenv').config();
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const mongoose = require('mongoose');
const Room = require('./models/Room');

// ─── Config ──────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
const MONGO_URI = process.env.MONGO_URI;

// ─── Express + Socket.io Setup ───────────────────────────────────────
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: '*' },
});

// ─── MongoDB Connection ──────────────────────────────────────────────
if (!MONGO_URI) {
  console.error('❌ FATAL ERROR: MONGO_URI is not defined in .env');
  process.exit(1);
}

mongoose.connect(MONGO_URI)
  .then(() => console.log('✅ Connected to MongoDB'))
  .catch(err => {
    console.error('❌ MongoDB Connection Error:', err);
    process.exit(1);
  });

// ─── Utility: Generate 6-char room code ──────────────────────────────
function generateRoomCode() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // no I/O/0/1 to avoid confusion
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

// ─── Socket.io Event Handling ────────────────────────────────────────
io.on('connection', (socket) => {
  console.log(`🔌 Connected: ${socket.id}`);

  // ── Create Room ──────────────────────────────────────────────────
  socket.on('create-room', async (callback) => {
    try {
      let code;
      let existing;
      do {
        code = generateRoomCode();
        existing = await Room.findOne({ code });
      } while (existing);

      const room = new Room({
        code,
        players: [{ socketId: socket.id, playerNumber: 1 }],
      });
      await room.save();

      socket.join(code);
      console.log(`🏠 Room ${code} created by ${socket.id}`);

      callback({ success: true, code, playerNumber: 1 });
    } catch (err) {
      console.error(err);
      callback({ success: false, error: 'Server error' });
    }
  });

  // ── Join Room ────────────────────────────────────────────────────
  socket.on('join-room', async (code, callback) => {
    try {
      code = code.toUpperCase().trim();
      const room = await Room.findOne({ code });

      if (!room) {
        return callback({ success: false, error: 'Room not found' });
      }
      if (room.status !== 'waiting') {
        return callback({ success: false, error: 'Game already in progress' });
      }
      if (room.players.length >= 2) {
        return callback({ success: false, error: 'Room is full' });
      }

      // Add player 2
      room.players.push({ socketId: socket.id, playerNumber: 2 });
      room.initBoard();
      await room.save();

      socket.join(code);
      console.log(`🎮 Player 2 (${socket.id}) joined room ${code}`);

      callback({ success: true, code, playerNumber: 2 });

      io.to(code).emit('game-start', {
        board: room.board,
        currentPlayer: room.currentPlayer,
        scores: room.scores,
        turnCount: room.turnCount,
      });
    } catch (err) {
      console.error(err);
      callback({ success: false, error: 'Server error' });
    }
  });

  // ── Make Move ────────────────────────────────────────────────────
  socket.on('make-move', async ({ code, row, col }, callback) => {
    try {
      const room = await Room.findOne({ code });
      if (!room) {
        return callback({ valid: false, reason: 'Room not found' });
      }

      const player = room.players.find(p => p.socketId === socket.id);
      if (!player) {
        return callback({ valid: false, reason: 'You are not in this room' });
      }

      const result = room.applyMove(row, col, player.playerNumber);
      if (!result.valid) {
        return callback(result);
      }

      await room.save();

      // Broadcast to both players
      io.to(code).emit('move-made', {
        row,
        col,
        player: player.playerNumber,
        board: result.board,
        scores: result.scores,
        currentPlayer: result.currentPlayer,
        turnCount: result.turnCount,
        winner: result.winner,
        explosionSteps: result.explosionSteps,
      });

      callback({ valid: true });
    } catch (err) {
      console.error(err);
      callback({ valid: false, reason: 'Server error' });
    }
  });

  // ── Disconnect ───────────────────────────────────────────────────
  socket.on('disconnect', async () => {
    console.log(`🔌 Disconnected: ${socket.id}`);

    try {
      const roomsWithPlayer = await Room.find({ 'players.socketId': socket.id });

      for (const room of roomsWithPlayer) {
        const playerIndex = room.players.findIndex(p => p.socketId === socket.id);
        if (playerIndex === -1) continue;

        if (room.status === 'waiting') {
          await Room.deleteOne({ _id: room._id });
          console.log(`🗑️  Deleted waiting room ${room.code}`);
        } else if (room.status === 'playing') {
          room.status = 'finished';
          const disconnectedPlayer = room.players[playerIndex];
          room.winner = disconnectedPlayer.playerNumber === 1 ? 2 : 1;
          await room.save();

          io.to(room.code).emit('opponent-disconnected', {
            winner: room.winner,
          });
          console.log(`💔 Player disconnected from room ${room.code}`);
        }
      }
    } catch (err) {
      console.error('Disconnect handling error:', err);
    }
  });
});

// ─── Start Server ────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n🎮 Color Wars backend server running at http://localhost:${PORT}\n`);
});
