"""
Pipeline orchestrator for the Billboard Cultural Mood Analysis pipeline.

Responsibilities:
  - Generate pipeline_run_id
  - Write run manifest (Schema 12, two appends per run)
  - Apply force_rerun flags and downstream invalidation
  - Execute stage sequence in locked order
  - Pass pipeline_run_id into every stage
  - Log stage timing at INFO
  - Handle per-stage exceptions without aborting the full run
"""
import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.core.checkpoint import invalidate_from
from src.core.config import PipelineConfig, load_config
from src.core.logger import get_logger, init_run_logging
from src.core.archiver import archive_run

log = get_logger("pipeline")


# ── Run manifest (Schema 12, locked pre-P5) ───────────────────────────────────

def generate_run_id(config: PipelineConfig) -> str:
    """
    Generate a deterministic pipeline_run_id for this run.

    Format (locked pre-P5): {decade}_{YYYYMMDD}T{HHMMSS}
    Example: 1990s_20240501T143200

    Decade is derived from config.project.year_range[0].

    Args:
        config: PipelineConfig instance

    Returns:
        Run ID string.
    """
    start_year = config.project.year_range[0]
    decade = f"{(start_year // 10) * 10}s"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{decade}_{ts}"


def _config_hash(config: PipelineConfig) -> str:
    """
    Compute a SHA-256 hash of the serialised config for run traceability.

    Uses the model_dump JSON representation. Returns first 16 hex chars.
    """
    raw = json.dumps(config.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def write_run_manifest(
    run_id: str,
    config: PipelineConfig,
    started_at: str,
    completed_at: Optional[str] = None,
    gate_pass: Optional[bool] = None,
) -> None:
    """
    Append one record to logs/run_manifest.jsonl.

    Called twice per run:
      1. At startup: completed_at=None, gate_pass=None
      2. At completion: completed_at and gate_pass populated

    The manifest is append-only — records are never mutated.
    Two records with the same run_id represent start and end of that run.
    A run with only a startup record did not complete.

    Args:
        run_id: pipeline_run_id string
        config: PipelineConfig instance
        started_at: ISO 8601 UTC timestamp of run start
        completed_at: ISO 8601 UTC timestamp of completion, or None
        gate_pass: final gate_pass value from S5, or None
    """
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "run_manifest.jsonl"

    record = {
        "pipeline_run_id": run_id,
        "year_range":       config.project.year_range,
        "decade":           f"{(config.project.year_range[0] // 10) * 10}s",
        "config_hash":      _config_hash(config),
        "config_path":      str(Path("config/config.yaml").resolve()),
        "started_at":       started_at,
        "completed_at":     completed_at,
        "gate_pass":        gate_pass,
        "python_version":   f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform":         platform.platform(terse=True),
    }

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Force-rerun application ───────────────────────────────────────────────────

def apply_force_reruns(config: PipelineConfig) -> None:
    """
    For each stage with force_rerun=True, delete that stage's checkpoint
    and all downstream checkpoints before the pipeline begins.

    Processes stages in dependency order so that if multiple force flags
    are set, the earliest one's invalidation subsumes the later ones.

    Uses the decade string derived from config for corpus checkpoint naming.

    Args:
        config: PipelineConfig instance
    """
    decade = f"{(config.project.year_range[0] // 10) * 10}s"
    fr = config.checkpoints.force_rerun

    stage_flags = [
        ("s1_ingest",    fr.s1_ingest),
        ("s2_lyrics",    fr.s2_lyrics),
        ("s3_sentiment", fr.s3_sentiment),
        ("s3_mood",      fr.s3_mood),
        ("s3_theme",     fr.s3_theme),
        ("s3_jungian",   fr.s3_jungian),
        ("s3_semantic",  fr.s3_semantic),
        ("s4_merge",     fr.s4_merge),
        ("s5_report",    fr.s5_report),
    ]

    for stage, is_set in stage_flags:
        if is_set:
            log.info(f"force_rerun.{stage}=True — invalidating from {stage}")
            invalidate_from(stage, config, decade=decade)
            # Once the earliest set flag is processed, downstream stages
            # are already invalidated — no need to process later flags
            # that are subsumed. Continue loop in case non-contiguous flags
            # are set (e.g. s3_sentiment and s5_report both True).


# ── Stage timing helper ───────────────────────────────────────────────────────

class _Timer:
    """Minimal context manager for logging stage elapsed time."""

    def __init__(self, stage_name: str):
        self._name = stage_name
        self._start: Optional[datetime] = None

    def __enter__(self):
        import time
        self._start = time.monotonic()
        log.info(f"Stage START: {self._name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            log.info(f"Stage END:   {self._name} ({elapsed:.1f}s)")
        else:
            log.error(f"Stage FAIL:  {self._name} ({elapsed:.1f}s) — {exc_val}")
        return False  # do not suppress exceptions


# ── Pipeline execution ────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig, run_id: str) -> Optional[bool]:
    """
    Execute all pipeline stages in the locked sequence.

    Stage sequence (locked P3 / P6):
      S1 → S2 → S3a → S3b → S3c (calls S3d internally) → S3e → S4 → S5

    Each stage receives the run_id so it can tag its outputs. Per-song
    exceptions within a stage are handled inside the stage itself. This
    function handles stage-level exceptions — a stage that raises is
    logged at ERROR and the pipeline continues to the next stage where
    possible (S4 and S5 require earlier outputs; they will fail cleanly
    if checkpoints are missing).

    Args:
        config: PipelineConfig instance
        run_id: pipeline_run_id string

    Returns:
        gate_pass boolean from S5, or None if S5 did not complete.
    """
    from src.stages import (
        s1_ingest,
        s2_lyrics,
        s3_jungian,
        s3_mood,
        s3_semantic,
        s3_sentiment,
        s3_theme,
        s4_merge,
        s5_report,
    )

    gate_pass: Optional[bool] = None

    # ── S1: Ingest ──
    songs_df = None
    try:
        with _Timer("S1 ingest"):
            songs_df = s1_ingest.run(config)
    except Exception:
        log.error(f"S1 ingest failed:\n{traceback.format_exc()}")

    # ── S2: Lyrics ──
    lyrics_df = None
    try:
        with _Timer("S2 lyrics"):
            lyrics_df = s2_lyrics.run(config, songs_df=songs_df, run_id=run_id)
    except Exception:
        log.error(f"S2 lyrics failed:\n{traceback.format_exc()}")

    # ── S3a: Sentiment ──
    try:
        with _Timer("S3a sentiment"):
            s3_sentiment.run(config, lyrics_df=lyrics_df, run_id=run_id)
    except Exception:
        log.error(f"S3a sentiment failed:\n{traceback.format_exc()}")

    # ── S3b: Mood ──
    try:
        with _Timer("S3b mood"):
            s3_mood.run(config, lyrics_df=lyrics_df, run_id=run_id)
    except Exception:
        log.error(f"S3b mood failed:\n{traceback.format_exc()}")

    # ── S3c: Theme (calls S3d Jungian internally for fallback) ──
    try:
        with _Timer("S3c theme"):
            s3_theme.run(config, lyrics_df=lyrics_df, run_id=run_id)
    except Exception:
        log.error(f"S3c theme failed:\n{traceback.format_exc()}")

    # ── S3d: Jungian (full pass, independent of theme fallback calls) ──
    try:
        with _Timer("S3d jungian"):
            s3_jungian.run(config, lyrics_df=lyrics_df, run_id=run_id)
    except Exception:
        log.error(f"S3d jungian failed:\n{traceback.format_exc()}")

    # ── S3e: Semantic ──
    try:
        with _Timer("S3e semantic"):
            s3_semantic.run(config, lyrics_df=lyrics_df, run_id=run_id)
    except Exception:
        log.error(f"S3e semantic failed:\n{traceback.format_exc()}")

    # ── S4: Merge ──
    merged_df = None
    try:
        with _Timer("S4 merge"):
            merged_df = s4_merge.run(config, run_id=run_id)
    except Exception:
        log.error(f"S4 merge failed:\n{traceback.format_exc()}")

    # ── S5: Report ──
    try:
        with _Timer("S5 report"):
            result = s5_report.run(config, merged_df=merged_df, run_id=run_id)
            gate_pass = result.get("gate_pass") if result else None
    except Exception:
        log.error(f"S5 report failed:\n{traceback.format_exc()}")

    return gate_pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main(config_path: str = "config/config.yaml") -> None:
    """
    Pipeline entry point.

    Loads config, initialises logging, generates run_id, writes startup
    manifest record, applies force reruns, runs all stages, writes
    completion manifest record.

    Args:
        config_path: path to config.yaml (default: config/config.yaml)
    """
    config = load_config(config_path)

    run_id = generate_run_id(config)
    started_at = datetime.now(timezone.utc).isoformat()

    # Initialise logging — must happen before any get_logger() call
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    init_run_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        run_timestamp=ts,
    )

    log.info(f"Pipeline starting. run_id={run_id} year_range={config.project.year_range}")

    # ── Startup manifest record ──
    write_run_manifest(
        run_id=run_id,
        config=config,
        started_at=started_at,
    )

    # ── Apply force reruns ──
    apply_force_reruns(config)

    # ── Run all stages ──
    gate_pass = run_pipeline(config=config, run_id=run_id)

    # ── Completion manifest record ──
    completed_at = datetime.now(timezone.utc).isoformat()
    write_run_manifest(
        run_id=run_id,
        config=config,
        started_at=started_at,
        completed_at=completed_at,
        gate_pass=gate_pass,
    )

    # ── Archive outputs ──
    archive_run(config=config, run_id=run_id)

    gate_str = "PASS" if gate_pass else ("FAIL" if gate_pass is False else "UNKNOWN")
    log.info(
        f"Pipeline complete. run_id={run_id} "
        f"gate={gate_str} "
        f"completed_at={completed_at}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Billboard Cultural Mood Analysis Pipeline"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
