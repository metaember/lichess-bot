"""SQLite game logger for the lichess-bot.

Writes game metadata, per-move engine evals, and live board state
to a SQLite database for the spectator web UI to read.
"""
from __future__ import annotations

import logging
import sqlite3
import datetime
from pathlib import Path
from typing import Any

import chess

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "games.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    lichess_url TEXT,
    bot_color TEXT,
    opponent_name TEXT,
    opponent_rating INTEGER,
    opponent_is_bot INTEGER,
    time_control TEXT,
    speed TEXT,
    mode TEXT,
    provenance TEXT,
    status TEXT DEFAULT 'playing',
    result TEXT DEFAULT '*',
    termination TEXT,
    started_at TEXT,
    finished_at TEXT,
    moves_uci TEXT
);

CREATE TABLE IF NOT EXISTS move_evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT REFERENCES games(game_id),
    ply INTEGER,
    move_uci TEXT,
    move_san TEXT,
    eval_cp INTEGER,
    eval_mate INTEGER,
    depth INTEGER,
    pv TEXT,
    nodes INTEGER,
    nps INTEGER,
    time_ms INTEGER,
    source TEXT,
    clock_ms INTEGER
);

CREATE TABLE IF NOT EXISTS live_state (
    game_id TEXT PRIMARY KEY REFERENCES games(game_id),
    fen TEXT,
    last_move_uci TEXT,
    moves_uci TEXT,
    wtime_ms INTEGER,
    btime_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_move_evals_game ON move_evals(game_id, ply);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_started ON games(started_at);
"""


class GameLogger:
    """Logs game state and engine evaluations to SQLite."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        # Migrate: add moves_uci column to games if missing (added after initial schema)
        try:
            self._conn.execute("ALTER TABLE games ADD COLUMN moves_uci TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        self._conn.commit()
        logger.info(f"GameLogger initialized: {db_path}")

    def close(self) -> None:
        self._conn.close()

    def game_started(self, game: Any, provenance: str) -> None:
        """Record a new game starting."""
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO games
                   (game_id, lichess_url, bot_color, opponent_name, opponent_rating,
                    opponent_is_bot, time_control, speed, mode, provenance,
                    status, result, started_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'playing', '*', ?)""",
                (
                    game.id,
                    game.short_url(),
                    game.my_color,
                    game.opponent.name,
                    game.opponent.rating,
                    1 if game.opponent.is_bot else 0,
                    game.time_control(),
                    game.speed,
                    game.mode,
                    provenance,
                    game.game_start.isoformat(),
                ),
            )
            self._conn.execute(
                """INSERT OR REPLACE INTO live_state
                   (game_id, fen, last_move_uci, moves_uci, wtime_ms, btime_ms)
                   VALUES (?, ?, NULL, '', ?, ?)""",
                (
                    game.id,
                    chess.STARTING_FEN,
                    int(game.clock_initial.total_seconds() * 1000),
                    int(game.clock_initial.total_seconds() * 1000),
                ),
            )
            self._conn.commit()
            logger.info(f"Game started: {game.id} ({provenance})")
        except Exception:
            logger.exception(f"Failed to log game start for {game.id}")

    def move_played(
        self,
        game: Any,
        board: chess.Board,
        move: chess.Move,
        commentary: dict[str, Any] | None,
        source: str = "search",
    ) -> None:
        """Record an engine move with its evaluation."""
        try:
            ply = len(board.move_stack) - 1  # board already has the move pushed

            # Get SAN from the position before the move was made
            board.pop()
            move_san = board.san(move)
            board.push(move)

            eval_cp = None
            eval_mate = None
            depth = None
            pv_str = None
            nodes = None
            nps = None
            time_ms = None

            if commentary:
                score = commentary.get("score")
                if score is not None:
                    # score is a PovScore from python-chess
                    white_score = score.white()
                    if white_score.is_mate():
                        eval_mate = white_score.mate()
                    else:
                        eval_cp = white_score.score()

                depth = commentary.get("depth")
                nodes = commentary.get("nodes")
                nps = commentary.get("nps")
                time_val = commentary.get("time")
                if time_val is not None:
                    time_ms = int(time_val * 1000) if isinstance(time_val, float) else int(time_val)
                pv_san = commentary.get("ponderpv")
                if pv_san:
                    pv_str = pv_san

            # Get bot's remaining clock
            clock_ms = None
            state = game.state
            if game.is_white:
                clock_ms = state.get("wtime")
            else:
                clock_ms = state.get("btime")

            self._conn.execute(
                """INSERT INTO move_evals
                   (game_id, ply, move_uci, move_san, eval_cp, eval_mate,
                    depth, pv, nodes, nps, time_ms, source, clock_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    game.id, ply, move.uci(), move_san, eval_cp, eval_mate,
                    depth, pv_str, nodes, nps, time_ms, source, clock_ms,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception(f"Failed to log move for {game.id}")

    def update_live_state(self, game: Any, board: chess.Board) -> None:
        """Update the live board state (called after every move, including opponent's)."""
        try:
            moves_uci = " ".join(m.uci() for m in board.move_stack)
            last_move = board.move_stack[-1].uci() if board.move_stack else None
            state = game.state

            self._conn.execute(
                """UPDATE live_state
                   SET fen = ?, last_move_uci = ?, moves_uci = ?,
                       wtime_ms = ?, btime_ms = ?
                   WHERE game_id = ?""",
                (
                    board.fen(),
                    last_move,
                    moves_uci,
                    state.get("wtime"),
                    state.get("btime"),
                    game.id,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception(f"Failed to update live state for {game.id}")

    def game_finished(self, game: Any) -> None:
        """Mark a game as finished and clean up live state."""
        try:
            winner = game.state.get("winner")
            termination = game.state.get("status")
            result = game.result()

            # Grab moves_uci from live_state before deleting it
            row = self._conn.execute(
                "SELECT moves_uci FROM live_state WHERE game_id = ?", (game.id,)
            ).fetchone()
            moves_uci = row[0] if row else None

            self._conn.execute(
                """UPDATE games
                   SET status = 'finished', result = ?, termination = ?,
                       finished_at = ?, moves_uci = ?
                   WHERE game_id = ?""",
                (
                    result,
                    termination,
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    moves_uci,
                    game.id,
                ),
            )
            self._conn.execute("DELETE FROM live_state WHERE game_id = ?", (game.id,))
            self._conn.commit()
            logger.info(f"Game finished: {game.id} result={result} term={termination}")
        except Exception:
            logger.exception(f"Failed to log game end for {game.id}")


# Module-level singleton
_logger: GameLogger | None = None


def get_game_logger() -> GameLogger:
    """Get or create the singleton GameLogger instance."""
    global _logger
    if _logger is None:
        _logger = GameLogger()
    return _logger
