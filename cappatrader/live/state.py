"""SQLite persistence so the paper account survives restarts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cappatrader.execution.account import Account
from cappatrader.models import Trade

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side INTEGER, qty REAL, entry_price REAL,
    exit_price REAL, entry_time TEXT, exit_time TEXT, pnl REAL, r_multiple REAL, exit_reason TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, equity REAL, cash REAL);
CREATE TABLE IF NOT EXISTS events (ts TEXT, kind TEXT, message TEXT);
"""


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


class StateStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def load_account(self, starting_balance: float) -> Account:
        row = self.conn.execute("SELECT value FROM kv WHERE key='account'").fetchone()
        if row is None:
            acct = Account(starting_balance)
            self.save_account(acct)
            return acct
        return Account.from_dict(json.loads(row[0]))

    def save_account(self, acct: Account) -> None:
        self.conn.execute("INSERT OR REPLACE INTO kv(key, value) VALUES('account', ?)",
                          (json.dumps(acct.to_dict(), default=_json_default),))
        self.conn.commit()

    def record_trade(self, t: Trade) -> None:
        self.conn.execute(
            "INSERT INTO trades(symbol, side, qty, entry_price, exit_price, entry_time, exit_time, pnl, r_multiple, "
            "exit_reason, data) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (t.symbol, int(t.side), float(t.qty), float(t.entry_price), float(t.exit_price), t.entry_time,
             t.exit_time, float(t.pnl), float(t.r_multiple), t.exit_reason,
             json.dumps(t.to_dict(), default=_json_default)),
        )
        self.conn.commit()

    def record_equity(self, ts: str, equity: float, cash: float) -> None:
        self.conn.execute("INSERT INTO equity VALUES (?,?,?)", (ts, float(equity), float(cash)))
        self.conn.commit()

    def record_event(self, ts: str, kind: str, message: str) -> None:
        self.conn.execute("INSERT INTO events VALUES (?,?,?)", (ts, kind, message))
        self.conn.commit()

    def trades(self, limit: int = 50) -> list[Trade]:
        rows = self.conn.execute("SELECT data FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Trade.from_dict(json.loads(r[0])) for r in reversed(rows)]

    def reset(self, starting_balance: float) -> Account:
        self.conn.executescript("DELETE FROM kv; DELETE FROM trades; DELETE FROM equity; DELETE FROM events;")
        acct = Account(starting_balance)
        self.save_account(acct)
        return acct
