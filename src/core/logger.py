import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Run-scoped timestamp ──────────────────────────────────────────────────────
# Set once at pipeline startup via init_run_logging(). All loggers share it.
_run_timestamp: Optional[str] = None
_log_dir: Optional[str] = None


# ── Log format ────────────────────────────────────────────────────────────────
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── Module-level logger registry ─────────────────────────────────────────────
_loggers: dict[str, logging.Logger] = {}


def init_run_logging(log_dir: str, level: str, run_timestamp: str) -> None:
    """
    Initialize the run-scoped logging environment.

    Must be called once at pipeline startup before any call to get_logger().
    Creates the log directory if it does not exist. Sets the run log file path
    using the provided timestamp.

    Args:
        log_dir: directory for log files (from config.logging.log_dir)
        level: logging level string e.g. "INFO" (from config.logging.level)
        run_timestamp: timestamp string used in log filename e.g. "20240501T143200"
    """
    global _run_timestamp, _log_dir
    _run_timestamp = run_timestamp
    _log_dir = log_dir

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"run_{run_timestamp}.log"

    root_logger = logging.getLogger("pipeline")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(_FORMATTER)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_FORMATTER)
    root_logger.addHandler(stream_handler)

    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a named child logger under the pipeline root logger.

    Returns the same instance on repeated calls with the same name.
    init_run_logging() must be called before this function is used in
    production. In tests, a fallback stderr logger is created automatically
    if init_run_logging() has not been called.

    Args:
        name: module or stage name e.g. "s1_ingest", "s2_lyrics"

    Returns:
        logging.Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    if _log_dir is None:
        # Fallback for tests that do not call init_run_logging
        logger = logging.getLogger(f"pipeline.{name}")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(_FORMATTER)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
    else:
        logger = logging.getLogger(f"pipeline.{name}")
        logger.propagate = True

    _loggers[name] = logger
    return logger


# ── JSONL append writers ──────────────────────────────────────────────────────

class _JsonlWriter:
    """
    Append-only writer for JSONL log files.
    Opens and closes the file on each write to avoid file handle leaks
    across long-running pipeline stages.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        """Append a single record as a JSON line."""
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class MissingLyricsWriter:
    """
    Writer for logs/missing_lyrics.jsonl.
    Each record must conform to Schema 9 (P5).
    """

    def __init__(self, path: str) -> None:
        self._writer = _JsonlWriter(path)

    def log(
        self,
        song_id: str,
        year: int,
        title: str,
        artist: str,
        genius_tried: bool,
        genius_error: Optional[str],
        musixmatch_tried: bool,
        musixmatch_error: Optional[str],
        pipeline_run_id: str,
    ) -> None:
        record = {
            "song_id": song_id,
            "year": year,
            "title": title,
            "artist": artist,
            "genius_tried": genius_tried,
            "genius_error": genius_error,
            "musixmatch_tried": musixmatch_tried,
            "musixmatch_error": musixmatch_error,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_run_id": pipeline_run_id,
        }
        self._writer.write(record)


class LowConfidenceWriter:
    """
    Writer for logs/low_confidence.jsonl.
    Each record must conform to Schema 10 (P5).
    """

    def __init__(self, path: str) -> None:
        self._writer = _JsonlWriter(path)

    def log(
        self,
        song_id: str,
        year: int,
        title: str,
        artist: str,
        dimension: str,
        flag_value: str,
        pipeline_run_id: str,
        score: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        allowed_dimensions = {"sentiment", "mood", "theme", "jungian"}
        if dimension not in allowed_dimensions:
            raise ValueError(
                f"LowConfidenceWriter: dimension must be one of {allowed_dimensions}, "
                f"got '{dimension}'"
            )
        record = {
            "song_id": song_id,
            "year": year,
            "title": title,
            "artist": artist,
            "dimension": dimension,
            "flag_value": flag_value,
            "score": score,
            "threshold": threshold,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_run_id": pipeline_run_id,
        }
        self._writer.write(record)


def get_missing_lyrics_logger(path: str) -> MissingLyricsWriter:
    """Return a MissingLyricsWriter for the given path."""
    return MissingLyricsWriter(path)


def get_low_confidence_logger(path: str) -> LowConfidenceWriter:
    """Return a LowConfidenceWriter for the given path."""
    return LowConfidenceWriter(path)
