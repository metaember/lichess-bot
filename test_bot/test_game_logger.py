"""Tests for the GameLogger SQLite game logger."""

from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from game_logger import GameLogger


# ---------------------------------------------------------------------------
# Mock objects that mimic model.Game / model.Player
# ---------------------------------------------------------------------------


@dataclass
class MockPlayer:
    name: str = "TestOpponent"
    rating: int = 1500
    is_bot: bool = False


@dataclass
class MockGame:
    id: str = "test_game_1"
    my_color: str = "white"
    is_white: bool = True
    speed: str = "blitz"
    mode: str = "rated"
    opponent: MockPlayer = field(default_factory=MockPlayer)
    game_start: datetime.datetime = field(
        default_factory=lambda: datetime.datetime(2025, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    )
    clock_initial: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(seconds=180))
    state: dict[str, Any] = field(
        default_factory=lambda: {"wtime": 180000, "btime": 180000, "status": "started", "moves": ""}
    )

    def short_url(self) -> str:
        return f"https://lichess.org/{self.id}"

    def time_control(self) -> str:
        return "180+2"

    def result(self) -> str:
        winner = self.state.get("winner")
        if winner == "white":
            return "1-0"
        elif winner == "black":
            return "0-1"
        return "*"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_games.db"


@pytest.fixture
def logger(db_path: Path) -> GameLogger:
    gl = GameLogger(db_path=db_path)
    yield gl
    gl.close()


@pytest.fixture
def conn(db_path: Path, logger: GameLogger) -> sqlite3.Connection:
    """Read-only connection for assertions."""
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def make_commentary(
    cp: int | None = 35,
    mate: int | None = None,
    depth: int = 12,
    nodes: int = 50000,
    nps: int = 1000000,
    time_sec: float = 0.05,
) -> dict[str, Any]:
    """Build a commentary dict mimicking engine_wrapper's InfoStrDict."""
    if mate is not None:
        score = chess.engine.PovScore(chess.engine.Mate(mate), chess.WHITE)
    elif cp is not None:
        score = chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)
    else:
        return {"depth": depth, "nodes": nodes, "nps": nps, "time": time_sec}

    return {
        "score": score,
        "depth": depth,
        "nodes": nodes,
        "nps": nps,
        "time": time_sec,
        "ponderpv": "e5 Nf3",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    def test_tables_exist(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "games" in tables
        assert "move_evals" in tables
        assert "live_state" in tables

    def test_wal_mode(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_indices_exist(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        indices = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        assert "idx_move_evals_game" in indices
        assert "idx_games_status" in indices
        assert "idx_games_started" in indices


class TestGameStarted:
    def test_inserts_game_row(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game.id,)).fetchone()
        assert row is not None
        assert row["game_id"] == "test_game_1"
        assert row["lichess_url"] == "https://lichess.org/test_game_1"
        assert row["bot_color"] == "white"
        assert row["opponent_name"] == "TestOpponent"
        assert row["opponent_rating"] == 1500
        assert row["opponent_is_bot"] == 0
        assert row["time_control"] == "180+2"
        assert row["speed"] == "blitz"
        assert row["mode"] == "rated"
        assert row["provenance"] == "matchmaking"
        assert row["status"] == "playing"
        assert row["result"] == "*"

    def test_inserts_live_state(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "incoming_challenge")

        row = conn.execute("SELECT * FROM live_state WHERE game_id = ?", (game.id,)).fetchone()
        assert row is not None
        assert row["fen"] == chess.STARTING_FEN
        assert row["last_move_uci"] is None
        assert row["moves_uci"] == ""
        assert row["wtime_ms"] == 180000
        assert row["btime_ms"] == 180000

    def test_bot_opponent(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame(opponent=MockPlayer(is_bot=True))
        logger.game_started(game, "incoming_challenge")

        row = conn.execute("SELECT opponent_is_bot FROM games WHERE game_id = ?", (game.id,)).fetchone()
        assert row["opponent_is_bot"] == 1


class TestMovePlayed:
    def test_with_commentary(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)
        commentary = make_commentary(cp=35, depth=12, nodes=50000, nps=1000000, time_sec=0.05)

        logger.move_played(game, board, move, commentary, source="search")

        row = conn.execute("SELECT * FROM move_evals WHERE game_id = ?", (game.id,)).fetchone()
        assert row is not None
        assert row["ply"] == 0
        assert row["move_uci"] == "e2e4"
        assert row["move_san"] == "e4"
        assert row["eval_cp"] == 35
        assert row["eval_mate"] is None
        assert row["depth"] == 12
        assert row["nodes"] == 50000
        assert row["nps"] == 1000000
        assert row["time_ms"] == 50
        assert row["source"] == "search"
        assert row["pv"] == "e5 Nf3"

    def test_without_commentary(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)

        logger.move_played(game, board, move, None, source="book")

        row = conn.execute("SELECT * FROM move_evals WHERE game_id = ?", (game.id,)).fetchone()
        assert row is not None
        assert row["eval_cp"] is None
        assert row["eval_mate"] is None
        assert row["depth"] is None
        assert row["source"] == "book"

    def test_with_mate_score(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)
        commentary = make_commentary(mate=3)

        logger.move_played(game, board, move, commentary, source="search")

        row = conn.execute("SELECT * FROM move_evals WHERE game_id = ?", (game.id,)).fetchone()
        assert row["eval_mate"] == 3
        assert row["eval_cp"] is None

    def test_negative_mate_score(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)
        commentary = make_commentary(mate=-5)

        logger.move_played(game, board, move, commentary, source="search")

        row = conn.execute("SELECT * FROM move_evals WHERE game_id = ?", (game.id,)).fetchone()
        assert row["eval_mate"] == -5

    def test_clock_tracking(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        game.state["wtime"] = 175000
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)

        logger.move_played(game, board, move, make_commentary(), source="search")

        row = conn.execute("SELECT clock_ms FROM move_evals WHERE game_id = ?", (game.id,)).fetchone()
        assert row["clock_ms"] == 175000

    def test_black_clock_tracking(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame(is_white=False, my_color="black")
        game.state["btime"] = 170000
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))  # white's move
        move = chess.Move.from_uci("e7e5")
        board.push(move)

        logger.move_played(game, board, move, make_commentary(), source="search")

        row = conn.execute("SELECT clock_ms FROM move_evals WHERE game_id = ?", (game.id,)).fetchone()
        assert row["clock_ms"] == 170000


class TestUpdateLiveState:
    def test_updates_fen_and_moves(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        game.state["wtime"] = 175000
        game.state["btime"] = 178000

        logger.update_live_state(game, board)

        row = conn.execute("SELECT * FROM live_state WHERE game_id = ?", (game.id,)).fetchone()
        assert row["fen"] == board.fen()
        assert row["last_move_uci"] == "e7e5"
        assert row["moves_uci"] == "e2e4 e7e5"
        assert row["wtime_ms"] == 175000
        assert row["btime_ms"] == 178000

    def test_single_move(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        board = chess.Board()
        board.push_uci("d2d4")

        logger.update_live_state(game, board)

        row = conn.execute("SELECT * FROM live_state WHERE game_id = ?", (game.id,)).fetchone()
        assert row["last_move_uci"] == "d2d4"
        assert row["moves_uci"] == "d2d4"


class TestGameFinished:
    def test_marks_finished(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")

        game.state["winner"] = "white"
        game.state["status"] = "mate"

        logger.game_finished(game)

        row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game.id,)).fetchone()
        assert row["status"] == "finished"
        assert row["result"] == "1-0"
        assert row["termination"] == "mate"
        assert row["finished_at"] is not None

    def test_deletes_live_state(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")
        logger.game_finished(game)

        row = conn.execute("SELECT * FROM live_state WHERE game_id = ?", (game.id,)).fetchone()
        assert row is None

    def test_black_wins(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame()
        logger.game_started(game, "matchmaking")
        game.state["winner"] = "black"
        game.state["status"] = "resign"
        logger.game_finished(game)

        row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game.id,)).fetchone()
        assert row["result"] == "0-1"
        assert row["termination"] == "resign"


class TestFullLifecycle:
    def test_complete_game(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game = MockGame(id="lifecycle_test")
        logger.game_started(game, "incoming_challenge")

        # Verify initial state
        assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM live_state").fetchone()[0] == 1

        # Play a few moves
        board = chess.Board()

        # Move 1: e4
        move1 = chess.Move.from_uci("e2e4")
        board.push(move1)
        logger.move_played(game, board, move1, make_commentary(cp=30), source="book")
        logger.update_live_state(game, board)

        # Opponent plays e5
        board.push_uci("e7e5")
        game.state["wtime"] = 177000
        game.state["btime"] = 178000
        logger.update_live_state(game, board)

        # Move 2: Nf3
        move2 = chess.Move.from_uci("g1f3")
        board.push(move2)
        logger.move_played(game, board, move2, make_commentary(cp=25, depth=14), source="search")
        logger.update_live_state(game, board)

        # Verify evals
        evals = conn.execute(
            "SELECT * FROM move_evals WHERE game_id = ? ORDER BY ply", (game.id,)
        ).fetchall()
        assert len(evals) == 2
        assert evals[0]["move_san"] == "e4"
        assert evals[1]["move_san"] == "Nf3"

        # Verify live state is up to date
        live = conn.execute("SELECT * FROM live_state WHERE game_id = ?", (game.id,)).fetchone()
        assert "e2e4 e7e5 g1f3" == live["moves_uci"]

        # Finish the game
        game.state["winner"] = "white"
        game.state["status"] = "mate"
        logger.game_finished(game)

        # Verify final state
        final = conn.execute("SELECT * FROM games WHERE game_id = ?", (game.id,)).fetchone()
        assert final["status"] == "finished"
        assert final["result"] == "1-0"

        # Live state cleaned up
        assert conn.execute("SELECT COUNT(*) FROM live_state").fetchone()[0] == 0


class TestErrorHandling:
    def test_game_started_bad_game_does_not_raise(self, logger: GameLogger) -> None:
        """GameLogger should catch exceptions, not crash the bot."""

        class BadGame:
            id = "bad"

        # This will fail because BadGame lacks required attributes,
        # but it should be caught internally
        logger.game_started(BadGame(), "test")  # type: ignore[arg-type]

    def test_move_played_bad_board_does_not_raise(self, logger: GameLogger) -> None:
        game = MockGame()
        logger.game_started(game, "test")

        # Pass a board with no moves (pop will fail)
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        # Don't push the move — move_played will try to pop and fail
        logger.move_played(game, board, move, None, source="search")

    def test_game_finished_nonexistent_does_not_raise(self, logger: GameLogger) -> None:
        game = MockGame(id="does_not_exist")
        logger.game_finished(game)


class TestMultipleGames:
    def test_concurrent_games(self, logger: GameLogger, conn: sqlite3.Connection) -> None:
        game1 = MockGame(id="game_1")
        game2 = MockGame(id="game_2", my_color="black", is_white=False)

        logger.game_started(game1, "matchmaking")
        logger.game_started(game2, "incoming_challenge")

        assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM live_state").fetchone()[0] == 2

        # Finish game1, game2 still live
        game1.state["winner"] = "white"
        game1.state["status"] = "mate"
        logger.game_finished(game1)

        assert conn.execute("SELECT COUNT(*) FROM live_state").fetchone()[0] == 1
        live = conn.execute("SELECT game_id FROM live_state").fetchone()
        assert live["game_id"] == "game_2"
