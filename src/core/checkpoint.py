import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.core.config import PipelineConfig
from src.core.logger import get_logger

log = get_logger("checkpoint")

# ── Stage dependency graph ────────────────────────────────────────────────────
# Maps each stage key to the ordered list of stages that follow it.
# Used to determine which checkpoints to invalidate on force rerun.
_STAGE_ORDER = [
    "s1_ingest",
    "s2_lyrics",
    "s3_sentiment",
    "s3_mood",
    "s3_theme",
    "s3_jungian",
    "s3_semantic",
    "s4_merge",
    "s5_report",
]

# Maps stage keys to their checkpoint filenames.
# s3_semantic has two checkpoints — corpus and parquet — both invalidated together.
_STAGE_FILES: dict[str, list[str]] = {
    "s1_ingest":    ["01_songs.parquet"],
    "s2_lyrics":    ["02_lyrics.parquet"],
    "s3_sentiment": ["03_sentiment.parquet"],
    "s3_mood":      ["03_mood.parquet"],
    "s3_theme":     ["03_theme.parquet"],
    "s3_jungian":   ["03_jungian.parquet"],
    "s3_semantic":  [],  # corpus pkl is decade-scoped; handled separately
    "s4_merge":     ["04_merged.parquet"],
    "s5_report":    [],  # outputs are not in checkpoints/; handled separately
}

# Output files invalidated when s5_report is force-rerun.
_OUTPUT_FILES = [
    "outputs/validation_report.md",
    "outputs/viz/sentiment_drift.png",
    "outputs/viz/mood_heatmap.png",
    "outputs/viz/theme_frequency.png",
    "outputs/viz/jungian_distribution.png",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _checkpoint_dir(config: PipelineConfig) -> Path:
    return Path(config.checkpoints.dir)


def _parquet_path(stage: str, config: PipelineConfig) -> Optional[Path]:
    files = _STAGE_FILES.get(stage, [])
    if not files:
        return None
    return _checkpoint_dir(config) / files[0]


# ── Public interface ──────────────────────────────────────────────────────────

def checkpoint_exists(stage: str, config: PipelineConfig) -> bool:
    """
    Return True if the checkpoint file for the given stage exists on disk.

    For s3_semantic, checks for the semantic parquet only (not the corpus pkl,
    which is checked separately via corpus_checkpoint_exists).

    Args:
        stage: stage key matching a key in _STAGE_FILES
        config: PipelineConfig instance

    Returns:
        bool
    """
    if stage == "s3_semantic":
        path = _checkpoint_dir(config) / "03_semantic.parquet"
        return path.exists()
    if stage == "s5_report":
        path = Path(config.outputs.dir) / config.outputs.report_filename
        return path.exists()
    path = _parquet_path(stage, config)
    if path is None:
        return False
    return path.exists()


def read_checkpoint(stage: str, config: PipelineConfig) -> pd.DataFrame:
    """
    Read and return the parquet checkpoint for the given stage.

    Args:
        stage: stage key
        config: PipelineConfig instance

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: if checkpoint does not exist
    """
    if stage == "s3_semantic":
        path = _checkpoint_dir(config) / "03_semantic.parquet"
    else:
        path = _parquet_path(stage, config)

    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Checkpoint for stage '{stage}' not found at expected path."
        )
    log.debug(f"Reading checkpoint: {path}")
    return pd.read_parquet(path)


def write_checkpoint(stage: str, df: pd.DataFrame, config: PipelineConfig) -> None:
    """
    Write a dataframe to the stage checkpoint file atomically.

    Writes to a .tmp file first, then renames to the final path.
    On APFS (macOS default), same-volume rename is atomic.
    A checkpoint file either exists and is complete, or does not exist.

    Args:
        stage: stage key
        df: dataframe to persist
        config: PipelineConfig instance

    Raises:
        ValueError: if stage has no associated parquet file
    """
    if stage == "s3_semantic":
        final_path = _checkpoint_dir(config) / "03_semantic.parquet"
    else:
        final_path = _parquet_path(stage, config)

    if final_path is None:
        raise ValueError(
            f"Stage '{stage}' has no parquet checkpoint. "
            f"Use write_corpus_checkpoint for corpus artifacts."
        )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(".parquet.tmp")

    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.rename(final_path)
        log.info(f"Checkpoint written: {final_path} ({len(df)} records)")
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_corpus_checkpoint(decade: str, vectorizer: Any, config: PipelineConfig) -> None:
    """
    Write the TF-IDF fitted vectorizer to a decade-scoped pickle checkpoint.

    Filename: checkpoints/03_tfidf_corpus_{decade}.pkl
    Locked in G2-R2 and G3 Recommendation 5.

    Args:
        decade: decade string from Schema 1 e.g. "1990s"
        vectorizer: fitted sklearn TfidfVectorizer instance
        config: PipelineConfig instance
    """
    checkpoint_dir = _checkpoint_dir(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"03_tfidf_corpus_{decade}.pkl"
    tmp_path = final_path.with_suffix(".pkl.tmp")

    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(vectorizer, f)
        tmp_path.rename(final_path)
        log.info(f"Corpus checkpoint written: {final_path}")
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def read_corpus_checkpoint(decade: str, config: PipelineConfig) -> Any:
    """
    Read and return the fitted TF-IDF vectorizer for the given decade.

    Args:
        decade: decade string e.g. "1990s"
        config: PipelineConfig instance

    Returns:
        Fitted sklearn TfidfVectorizer

    Raises:
        FileNotFoundError: if corpus checkpoint does not exist
    """
    path = _checkpoint_dir(config) / f"03_tfidf_corpus_{decade}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus checkpoint not found: {path}"
        )
    log.debug(f"Reading corpus checkpoint: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def corpus_checkpoint_exists(decade: str, config: PipelineConfig) -> bool:
    """Return True if the corpus pkl checkpoint exists for the given decade."""
    path = _checkpoint_dir(config) / f"03_tfidf_corpus_{decade}.pkl"
    return path.exists()


def get_downstream_stages(stage: str) -> list[str]:
    """
    Return the ordered list of stages that follow the given stage.

    Args:
        stage: stage key

    Returns:
        List of stage keys that come after stage in _STAGE_ORDER.

    Raises:
        ValueError: if stage is not a known stage key.
    """
    if stage not in _STAGE_ORDER:
        raise ValueError(
            f"Unknown stage '{stage}'. Known stages: {_STAGE_ORDER}"
        )
    idx = _STAGE_ORDER.index(stage)
    return _STAGE_ORDER[idx + 1:]


def invalidate_from(stage: str, config: PipelineConfig, decade: Optional[str] = None) -> None:
    """
    Delete checkpoints for the given stage and all downstream stages.

    For s3_semantic, also deletes the decade-scoped corpus pkl if decade is provided.
    For s5_report, deletes output files instead of checkpoint files.

    Args:
        stage: stage key to start invalidation from (inclusive)
        config: PipelineConfig instance
        decade: optional decade string required to invalidate corpus checkpoint

    Raises:
        ValueError: if stage is not a known stage key.
    """
    stages_to_clear = [stage] + get_downstream_stages(stage)
    log.warning(
        f"Invalidating checkpoints for stages: {stages_to_clear}"
    )

    for s in stages_to_clear:
        if s == "s3_semantic":
            parquet = _checkpoint_dir(config) / "03_semantic.parquet"
            if parquet.exists():
                parquet.unlink()
                log.warning(f"Deleted checkpoint: {parquet}")
            if decade:
                corpus = _checkpoint_dir(config) / f"03_tfidf_corpus_{decade}.pkl"
                if corpus.exists():
                    corpus.unlink()
                    log.warning(f"Deleted corpus checkpoint: {corpus}")
        elif s == "s5_report":
            for rel_path in _OUTPUT_FILES:
                p = Path(rel_path)
                if p.exists():
                    p.unlink()
                    log.warning(f"Deleted output: {p}")
            # Also remove the analysis NDJSON if present
            import glob
            for ndjson in glob.glob(f"{config.outputs.dir}/analysis_*.json"):
                Path(ndjson).unlink()
                log.warning(f"Deleted output: {ndjson}")
        else:
            path = _parquet_path(s, config)
            if path and path.exists():
                path.unlink()
                log.warning(f"Deleted checkpoint: {path}")
