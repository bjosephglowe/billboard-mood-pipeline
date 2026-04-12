"""
Artifact archiver for the Billboard Cultural Mood Analysis pipeline.

Responsibilities:
  - Copy all pipeline output artifacts to a local timestamped archive directory
  - Upload artifacts to GCS when configured
  - Enforce local retention policy (keep N most recent runs)
  - Fail softly — all errors are logged, none are re-raised

Two-layer strategy:
  Layer 1 (always): local copy at archive/{run_id}/
  Layer 2 (optional): GCS upload at gs://{bucket}/{prefix}/{run_id}/

GCS is skipped gracefully when:
  - archiving.gcs_enabled is False in config
  - GCS_BUCKET or GCS_KEY_PATH env vars are not set
  - google-cloud-storage is not installed

Local archiving always runs when archiving.enabled is True.
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from src.core.config import PipelineConfig
from src.core.logger import get_logger

log = get_logger("archiver")


# ── Artifact discovery ────────────────────────────────────────────────────────

def _collect_artifacts(config: PipelineConfig) -> list[Path]:
    """
    Collect all output artifacts produced by the current run.

    Looks in config.outputs.dir for:
      - validation_report.md
      - analysis_*.json (NDJSON output)
      - viz/*.png

    Returns a list of Path objects for files that exist. Missing files
    are logged at WARNING but do not raise.

    Args:
        config: PipelineConfig instance

    Returns:
        List of existing artifact paths.
    """
    outputs_dir = Path(config.outputs.dir)
    viz_dir = Path(config.outputs.viz_dir)

    candidates: list[Path] = []

    # Validation report
    report_path = outputs_dir / config.outputs.report_filename
    candidates.append(report_path)

    # Analysis NDJSON — glob for any analysis_*.json
    candidates.extend(outputs_dir.glob("analysis_*.json"))

    # Visualisations — glob for all PNGs in viz dir
    if viz_dir.exists():
        candidates.extend(viz_dir.glob("*.png"))

    existing = [p for p in candidates if p.exists()]
    missing = [p for p in candidates if not p.exists()]

    for p in missing:
        log.warning(f"Archiver: expected artifact not found, skipping: {p}")

    log.info(f"Archiver: {len(existing)} artifacts collected for archiving")
    return existing


# ── Local archive ─────────────────────────────────────────────────────────────

def _archive_local(
    artifacts: list[Path],
    run_id: str,
    config: PipelineConfig,
) -> Optional[Path]:
    """
    Copy artifacts into archive/{run_id}/, preserving sub-directory structure.

    The viz/ subdirectory is preserved under the run archive:
      archive/{run_id}/validation_report.md
      archive/{run_id}/analysis_2010s.json
      archive/{run_id}/viz/sentiment_drift.png
      ...

    Args:
        artifacts: list of existing artifact paths
        run_id: pipeline_run_id string
        config: PipelineConfig instance

    Returns:
        Path to the created run archive directory, or None on failure.
    """
    archive_root = Path(config.archiving.local_archive_dir)
    run_archive_dir = archive_root / run_id

    try:
        run_archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Archiver: failed to create local archive dir {run_archive_dir}: {e}")
        return None

    outputs_dir = Path(config.outputs.dir)
    copied = 0
    failed = 0

    for artifact in artifacts:
        try:
            # Preserve relative path from outputs_dir
            relative = artifact.relative_to(outputs_dir)
            dest = run_archive_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact, dest)
            copied += 1
        except Exception as e:
            log.error(f"Archiver: failed to copy {artifact} to local archive: {e}")
            failed += 1

    log.info(
        f"Archiver: local archive complete — "
        f"{copied} copied, {failed} failed → {run_archive_dir}"
    )
    return run_archive_dir if copied > 0 else None


# ── Local retention policy ────────────────────────────────────────────────────

def _apply_local_retention(config: PipelineConfig) -> None:
    """
    Enforce local_retention_max_runs by deleting oldest run archive dirs.

    Retention is based on directory modification time (mtime). The N most
    recently modified run dirs are kept; older ones are removed.

    Skipped when local_retention_max_runs is 0 (keep all).

    Args:
        config: PipelineConfig instance
    """
    max_runs = config.archiving.local_retention_max_runs
    if max_runs == 0:
        return

    archive_root = Path(config.archiving.local_archive_dir)
    if not archive_root.exists():
        return

    run_dirs = sorted(
        [d for d in archive_root.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )

    to_delete = run_dirs[: max(0, len(run_dirs) - max_runs)]

    for d in to_delete:
        try:
            shutil.rmtree(d)
            log.info(f"Archiver: retention policy removed old archive: {d.name}")
        except Exception as e:
            log.error(f"Archiver: failed to remove old archive {d}: {e}")


# ── GCS upload ────────────────────────────────────────────────────────────────

def _upload_to_gcs(
    artifacts: list[Path],
    run_id: str,
    config: PipelineConfig,
) -> bool:
    """
    Upload artifacts to GCS at gs://{bucket}/{prefix}/{run_id}/.

    Requires:
      - GCS_BUCKET env var set
      - GCS_KEY_PATH env var set and file exists
      - google-cloud-storage installed

    Sub-directory structure is preserved relative to outputs_dir, matching
    the local archive layout.

    Args:
        artifacts: list of existing artifact paths
        run_id: pipeline_run_id string
        config: PipelineConfig instance

    Returns:
        True if all uploads succeeded, False if any failed or GCS is
        unavailable. Caller treats False as a soft warning.
    """
    # ── Env var checks ──
    bucket_name = os.environ.get("GCS_BUCKET", "").strip()
    key_path = os.environ.get("GCS_KEY_PATH", "").strip()

    if not bucket_name:
        log.warning("Archiver: GCS_BUCKET not set — skipping GCS upload")
        return False

    if not key_path:
        log.warning("Archiver: GCS_KEY_PATH not set — skipping GCS upload")
        return False

    if not Path(key_path).exists():
        log.error(f"Archiver: GCS_KEY_PATH file not found: {key_path} — skipping GCS upload")
        return False

    # ── Import guard ──
    try:
        from google.cloud import storage as gcs
    except ImportError:
        log.error(
            "Archiver: google-cloud-storage not installed — skipping GCS upload. "
            "Run: pip install google-cloud-storage==2.16.0"
        )
        return False

    # ── Client initialisation ──
    try:
        client = gcs.Client.from_service_account_json(key_path)
        bucket = client.bucket(bucket_name)
    except Exception as e:
        log.error(f"Archiver: GCS client init failed: {e}")
        return False

    outputs_dir = Path(config.outputs.dir)
    prefix = config.archiving.gcs_prefix
    uploaded = 0
    failed = 0

    for artifact in artifacts:
        try:
            relative = artifact.relative_to(outputs_dir)
            blob_name = f"{prefix}/{run_id}/{relative}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(artifact))
            uploaded += 1
        except Exception as e:
            log.error(f"Archiver: GCS upload failed for {artifact.name}: {e}")
            failed += 1

    log.info(
        f"Archiver: GCS upload complete — "
        f"{uploaded} uploaded, {failed} failed → "
        f"gs://{bucket_name}/{prefix}/{run_id}/"
    )
    return failed == 0


# ── Public interface ──────────────────────────────────────────────────────────

def archive_run(config: PipelineConfig, run_id: str) -> None:
    """
    Archive all output artifacts for a completed pipeline run.

    Executes both archiving layers in sequence. All exceptions are caught
    and logged — this function never raises.

    Layer 1: local copy to archive/{run_id}/
    Layer 2: GCS upload (when gcs_enabled=True and credentials present)

    Retention policy is applied after local archiving.

    Args:
        config: PipelineConfig instance
        run_id: pipeline_run_id string
    """
    if not config.archiving.enabled:
        log.info("Archiver: archiving.enabled=False — skipping")
        return

    log.info(f"Archiver: starting archive for run_id={run_id}")

    # ── Collect artifacts ──
    try:
        artifacts = _collect_artifacts(config)
    except Exception as e:
        log.error(f"Archiver: artifact collection failed: {e}")
        return

    if not artifacts:
        log.warning("Archiver: no artifacts found — nothing to archive")
        return

    # ── Layer 1: local archive ──
    try:
        _archive_local(artifacts, run_id, config)
    except Exception as e:
        log.error(f"Archiver: local archive raised unexpectedly: {e}")

    # ── Local retention policy ──
    try:
        _apply_local_retention(config)
    except Exception as e:
        log.error(f"Archiver: retention policy raised unexpectedly: {e}")

    # ── Layer 2: GCS upload ──
    if config.archiving.gcs_enabled:
        try:
            _upload_to_gcs(artifacts, run_id, config)
        except Exception as e:
            log.error(f"Archiver: GCS upload raised unexpectedly: {e}")
    else:
        log.info("Archiver: archiving.gcs_enabled=False — skipping GCS upload")
