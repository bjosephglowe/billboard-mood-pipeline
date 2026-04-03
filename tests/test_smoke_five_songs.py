"""
Shallow end-to-end integration test — five fixture songs, no live APIs,
no model weight downloads.

Verifies that the full pipeline from S1 through S4 executes without error
on a minimal fixture dataset and produces a merged parquet containing five
records with all Schema 8 fields present.

S5 (report/viz) is excluded from this test to avoid matplotlib rendering
overhead in CI-style runs. Schema 8 field presence is the gate criterion.

All external dependencies are mocked:
  - Genius and Musixmatch API calls return fixture lyrics
  - Transformer model inference returns deterministic fixed outputs
  - Claude Haiku API calls return deterministic fixed JSON

Run with:
    python -m pytest tests/test_smoke_five_songs.py -v
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.stages.s4_merge import _SCHEMA_8_COLUMNS

# ── Fixture data paths ────────────────────────────────────────────────────────
_FIXTURE_DATASET = "tests/fixtures/data/billboard_top100.json"
_FIXTURE_LYRICS = (
    "Wise men say only fools rush in\n"
    "But I can't help falling in love with you\n"
    "Shall I stay would it be a sin\n"
    "If I can't help falling in love with you\n"
    "Like a river flows surely to the sea\n"
    "Darling so it goes some things are meant to be\n"
    "Take my hand take my whole life too\n"
    "For I can't help falling in love with you\n"
)

# ── Mock helpers ──────────────────────────────────────────────────────────────

def _sentiment_pipe(texts, truncation=True, max_length=512):
    """Fixed sentiment output — positive at 0.75."""
    return [
        [
            {"label": "positive", "score": 0.75},
            {"label": "neutral",  "score": 0.15},
            {"label": "negative", "score": 0.10},
        ]
        for _ in texts
    ]


def _mood_pipe(texts, truncation=True, max_length=512):
    """Fixed mood output — joy at 0.70."""
    return [
        [
            {"label": "joy",      "score": 0.70},
            {"label": "sadness",  "score": 0.12},
            {"label": "anger",    "score": 0.06},
            {"label": "fear",     "score": 0.05},
            {"label": "disgust",  "score": 0.04},
            {"label": "surprise", "score": 0.02},
            {"label": "neutral",  "score": 0.01},
        ]
        for _ in texts
    ]


def _theme_pipe(text, candidate_labels, truncation=True, max_length=512):
    """Fixed theme output — love_and_romance at 0.80."""
    scores = [0.80 if l == "love_and_romance" else 0.02 for l in candidate_labels]
    return {"labels": candidate_labels, "scores": scores}


def _haiku_response():
    """Fixed Haiku JSON response for Jungian + theme fallback."""
    return json.dumps({
        "jungian": {
            "primary": "anima_animus",
            "secondary": None,
            "confidence": "high",
            "evidence": ["take my hand take my whole life too"],
            "flag": None,
        },
        "theme_fallback": {
            "requested": False,
            "primary": None,
            "primary_confidence": 0.0,
            "secondary": None,
            "secondary_confidence": 0.0,
        },
    })


# ── Config fixture ────────────────────────────────────────────────────────────

@pytest.fixture()
def smoke_config(minimal_config, tmp_path):
    """
    Config wired for the five-song smoke run.

    - Points dataset at fixture JSON
    - Disables all external caches
    - Uses tmp_path for all checkpoints, logs, outputs
    - Sets year_range to 1993 only (our five fixture songs)
    - Forces all reruns to ensure clean state
    """
    minimal_config.project.year_range = [1993, 1993]
    minimal_config.project.top_n = 10
    minimal_config.dataset.local_path = _FIXTURE_DATASET

    minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
    minimal_config.cache.lyrics_dir = str(tmp_path / "cache" / "lyrics")
    minimal_config.cache.inference_dir = str(tmp_path / "cache" / "inference")
    minimal_config.logging.log_dir = str(tmp_path / "logs")
    minimal_config.logging.missing_lyrics_log = str(
        tmp_path / "logs" / "missing_lyrics.jsonl"
    )
    minimal_config.logging.low_confidence_log = str(
        tmp_path / "logs" / "low_confidence.jsonl"
    )
    minimal_config.outputs.dir = str(tmp_path / "outputs")
    minimal_config.outputs.viz_dir = str(tmp_path / "outputs" / "viz")

    minimal_config.lyrics.cache_enabled = False
    minimal_config.lyrics.genius_sleep_time = 0.0
    minimal_config.inference.cache_enabled = False
    minimal_config.inference.batch_size = 5
    minimal_config.inference.sleep_between_batches = 0.0
    minimal_config.semantic.mtld_min_tokens = 5
    minimal_config.semantic.tfidf_top_k_keywords = 5
    minimal_config.semantic.subject_focus_min_pronouns = 2
    minimal_config.semantic.tfidf_max_features = 50
    minimal_config.theme.haiku_fallback_enabled = False
    minimal_config.jungian.max_retries = 1
    minimal_config.jungian.retry_sleep = 0.0

    minimal_config.genius_api_token = "fake_genius_token"
    minimal_config.musixmatch_api_key = "fake_mxm_key"
    minimal_config.anthropic_api_key = "fake_anthropic_key"

    # Force all stages to rerun from scratch
    for attr in [
        "s1_ingest", "s2_lyrics", "s3_sentiment", "s3_mood",
        "s3_theme", "s3_jungian", "s3_semantic", "s4_merge",
    ]:
        setattr(minimal_config.checkpoints.force_rerun, attr, True)

    for d in [
        tmp_path / "checkpoints",
        tmp_path / "cache" / "lyrics",
        tmp_path / "cache" / "inference",
        tmp_path / "logs",
        tmp_path / "outputs" / "viz",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return minimal_config


# ── Smoke test ────────────────────────────────────────────────────────────────

def test_full_pipeline_five_songs(smoke_config):
    """
    Run S1 through S4 on five fixture songs with all external dependencies
    mocked. Assert that:
      1. 04_merged.parquet is written to the checkpoint directory
      2. The merged DataFrame contains exactly five records
      3. All Schema 8 columns are present in the output
      4. Every record has a non-null song_id
      5. pipeline_run_id is set on all records
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
    )

    run_id = "1990s_smoke_test"

    # ── S1: ingest from fixture dataset ──
    songs_df = s1_ingest.run(smoke_config)
    assert len(songs_df) == 5, (
        f"Expected 5 songs from fixture dataset, got {len(songs_df)}"
    )

    # ── S2: lyrics — mock Genius to return fixture lyrics for all songs ──
    with patch(
        "src.stages.s2_lyrics.fetch_genius",
        return_value=(_FIXTURE_LYRICS, None),
    ):
        lyrics_df = s2_lyrics.run(
            smoke_config, songs_df=songs_df, run_id=run_id
        )

    assert len(lyrics_df) == 5
    assert (lyrics_df["lyrics_status"] == "found").all(), (
        "All five songs should have found lyrics"
    )

    # ── S3a: sentiment — mock model ──
    with patch(
        "src.stages.s3_sentiment.load_sentiment_model",
        return_value=(_sentiment_pipe, "cpu"),
    ):
        with patch("src.stages.s3_sentiment.unload_model"):
            s3_sentiment.run(
                smoke_config, lyrics_df=lyrics_df, run_id=run_id
            )

    # ── S3b: mood — mock model ──
    with patch(
        "src.stages.s3_mood.load_mood_model",
        return_value=(_mood_pipe, "cpu"),
    ):
        with patch("src.stages.s3_mood.unload_model"):
            s3_mood.run(
                smoke_config, lyrics_df=lyrics_df, run_id=run_id
            )

    # ── S3c: theme — mock model ──
    with patch(
        "src.stages.s3_theme.load_theme_model",
        return_value=(_theme_pipe, "cpu"),
    ):
        with patch("src.stages.s3_theme.unload_model"):
            s3_theme.run(
                smoke_config, lyrics_df=lyrics_df, run_id=run_id
            )

    # ── S3d: jungian — mock Haiku API ──
    with patch(
        "src.stages.s3_jungian.call_haiku",
        return_value=_haiku_response(),
    ):
        s3_jungian.run(
            smoke_config, lyrics_df=lyrics_df, run_id=run_id
        )

    # ── S3e: semantic — no mocking needed (CPU-only, no model) ──
    s3_semantic.run(
        smoke_config, lyrics_df=lyrics_df, run_id=run_id
    )

    # ── S4: merge ──
    merged_df = s4_merge.run(smoke_config, run_id=run_id)

    # ── Assertions ──

    # 1. Checkpoint file written
    checkpoint = Path(smoke_config.checkpoints.dir) / "04_merged.parquet"
    assert checkpoint.exists(), "04_merged.parquet was not written"

    # 2. Exactly five records
    assert len(merged_df) == 5, (
        f"Expected 5 merged records, got {len(merged_df)}"
    )

    # 3. All Schema 8 columns present
    for col in _SCHEMA_8_COLUMNS:
        assert col in merged_df.columns, f"Missing Schema 8 column: {col}"

    # 4. Every record has a non-null song_id
    assert merged_df["song_id"].notna().all(), (
        "All records must have a non-null song_id"
    )

    # 5. pipeline_run_id set on all records
    assert (merged_df["pipeline_run_id"] == run_id).all(), (
        "pipeline_run_id must be set on all records"
    )

    # 6. All records have found lyrics
    assert (merged_df["lyrics_status"] == "found").all()

    # 7. Sentiment scored for all records
    assert merged_df["sentiment_score"].notna().all(), (
        "All five songs should have sentiment scores"
    )

    # 8. Mood primary set for all records
    assert merged_df["mood_primary"].notna().all()

    # 9. Theme primary set for all records (not uncertain)
    assert merged_df["theme_primary"].notna().all()
    assert (merged_df["theme_primary"] != "uncertain").all()

    # 10. record_complete True for all five songs
    assert merged_df["record_complete"].all(), (
        "All five fixture songs should be record_complete=True"
    )
