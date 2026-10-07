"""Logging to console + rotating file, with secrets masked out of every record."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from cappatrader.config import Config


class SecretFilter(logging.Filter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = [s for s in secrets if s and len(s) >= 8]

    def filter(self, record: logging.LogRecord) -> bool:
        if self.secrets:
            text = record.getMessage()
            masked = text
            for s in self.secrets:
                masked = masked.replace(s, "***")
            if masked != text:
                record.msg, record.args = masked, None
        return True


def setup_logging(cfg: Config, verbose: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    secret_filter = SecretFilter([cfg.secrets.binance_api_key, cfg.secrets.binance_secret_key,
                                  cfg.secrets.telegram_token])
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(secret_filter)
    root.addHandler(console)
    try:
        Path(cfg.runtime.log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(Path(cfg.runtime.log_dir) / "cappatrader.log",
                                                  maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(secret_filter)
        root.addHandler(fh)
    except OSError:
        pass
    for noisy in ("aiohttp", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
