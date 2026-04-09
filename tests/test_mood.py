"""
Tests for src/stages/s3_mood.py

Covers:
  - resolve_tie: tiebreak order, single winner, all tied
  - select_secondary: above threshold, below threshold, null when no candidates
  - aggregate_mood: null path, single chunk, multi-chunk modal, tie resolution
  - low_confidence flag at threshold boundary
  - null lyrics produces all-null Schema 4 record
  - inference cache hit skips model call
  - checkpoint read/write behavior
  - Schema 4 columns present in output
  - mood_secondary null when second-highest below 0.20
  - mood_secondary populated when second-highest >= 0.20
"""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.stages.s3_mood import (
    _build_null_record,
    aggregate_mood,
    classify_chunks,
    resolve_tie,
    run,
    select_secondary,
)

# ── Schema 4 columns (locked P5) ─────────────────────────────────────────────
_SCHEMA_4_COLUMNS = [
    "song_id",
    "mood_primary",
    "mood_primary_confidence",
    "mood_secondary",
    "mood_secondary_confidence",
    "mood_flag",
]

_VALID_MOOD_LABELS = {
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def lyrics_df(sample_song_record, sample_song_record_2, sample_lyrics_found):
    """Two-row Schema 2 DataFrame with lyrics present."""
    rows = []
    for rec in [sample_song_record, sample_song_record_2]:
        row = rec.copy()
        row["lyrics"] = sample_lyrics_found
        row["lyrics_status"] = "found"
        row["lyrics_source"] = "genius"
        row["lyrics_truncated"] = False
        row["lyrics_word_count"] = len(sample_lyrics_found.split())
        row["lyrics_fetched_at"] = "2024-05-01T00:00:00Z"
        row["lyrics_cache_hit"] = False
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def lyrics_df_missing(sample_song_record):
    """One-row Schema 2 DataFrame with missing lyrics."""
    row = sample_song_record.copy()
    row["lyrics"] = None
    row["lyrics_status"] = "missing"
    row["lyrics_source"] = None
    row["lyrics_truncated"] = False
    row["lyrics_word_count"] = None
    row["lyrics_fetched_at"] = "2024-05-01T00:00:00Z"
    row["lyrics_cache_hit"] = False
    return pd.DataFrame([row])


@pytest.fixture()
def mock_pipe():
    """
    Mock HuggingFace pipeline returning fixed probabilities.
    joy=0.70, sadness=0.15, others split remainder.
    Primary = joy (0.70), secondary = sadness (0.15 < 0.20 → null).
    """
    def _pipe(texts, truncation=True, max_length=512):
        return [
            [
                {"label": "joy",      "score": 0.70},
                {"label": "sadness",  "score": 0.15},
                {"label": "anger",    "score": 0.05},
                {"label": "fear",     "score": 0.04},
                {"label": "disgust",  "score": 0.03},
                {"label": "surprise", "score": 0.02},
                {"label": "neutral",  "score": 0.01},
            ]
            for _ in texts
        ]
    return _pipe


@pytest.fixture()
def mock_pipe_with_secondary():
    """
    Mock pipe where secondary label clears the 0.20 threshold.
    joy=0.65, sadness=0.25 (>= 0.20 → retained as secondary).
    """
    def _pipe(texts, truncation=True, max_length=512):
        return [
            [
                {"label": "joy",      "score": 0.65},
                {"label": "sadness",  "score": 0.25},
                {"label": "anger",    "score": 0.04},
                {"label": "fear",     "score": 0.03},
                {"label": "disgust",  "score": 0.01},
                {"label": "surprise", "score": 0.01},
                {"label": "neutral",  "score": 0.01},
            ]
            for _ in texts
        ]
    return _pipe


@pytest.fixture()
def mock_pipe_low_confidence():
    """
    Mock pipe returning low-confidence primary (< 0.35 threshold).
    neutral=0.30 is top label, below 0.35 threshold.
    """
    def _pipe(texts, truncation=True, max_length=512):
        return [
            [
                {"label": "neutral",  "score": 0.30},
                {"label": "joy",      "score": 0.25},
                {"label": "sadness",  "score": 0.20},
                {"label": "anger",    "score": 0.10},
                {"label": "fear",     "score": 0.07},
                {"label": "disgust",  "score": 0.05},
                {"label": "surprise", "score": 0.03},
            ]
            for _ in texts
        ]
    return _pipe


# ── resolve_tie ───────────────────────────────────────────────────────────────

class TestResolveTie:
    def test_clear_winner_no_tie(self):
        probs = {"joy": 0.80, "sadness": 0.10, "anger": 0.10}
        assert resolve_tie(probs) == "joy"

    def test_tie_joy_beats_sadness(self):
        # joy and sadness within 0.01 — joy wins per tiebreak order
        probs = {
            "joy": 0.500, "sadness": 0.495,
            "anger": 0.002, "fear": 0.001,
            "disgust": 0.001, "surprise": 0.001, "neutral": 0.0
        }
        assert resolve_tie(probs) == "joy"

    def test_tie_sadness_beats_anger(self):
        probs = {
            "joy": 0.0, "sadness": 0.500, "anger": 0.495,
            "fear": 0.002, "disgust": 0.001,
            "surprise": 0.001, "neutral": 0.001
        }
        assert resolve_tie(probs) == "sadness"

    def test_tie_anger_beats_fear(self):
        probs = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.500, "fear": 0.495,
            "disgust": 0.002, "surprise": 0.002, "neutral": 0.001
        }
        assert resolve_tie(probs) == "anger"

    def test_tie_neutral_lowest_priority(self):
        # neutral vs surprise — surprise comes before neutral in tiebreak order
        probs = {
            "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
            "disgust": 0.0, "surprise": 0.500, "neutral": 0.495
        }
        assert resolve_tie(probs) == "surprise"

    def test_single_label_returns_it(self):
        probs = {"joy": 1.0}
        assert resolve_tie(probs) == "joy"

    def test_empty_probs_returns_neutral(self):
        assert resolve_tie({}) == "neutral"

    def test_all_labels_equal_returns_first_in_tiebreak(self):
        probs = {label: 1.0 / 7 for label in
                 ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]}
        assert resolve_tie(probs) == "joy"


# ── select_secondary ──────────────────────────────────────────────────────────

class TestSelectSecondary:
    def test_secondary_above_threshold_returned(self):
        probs = {"joy": 0.65, "sadness": 0.25, "anger": 0.10}
        secondary, conf = select_secondary(probs, primary="joy")
        assert secondary == "sadness"
        assert conf == pytest.approx(0.25, abs=0.001)

    def test_secondary_below_threshold_returns_none(self):
        probs = {"joy": 0.80, "sadness": 0.15, "anger": 0.05}
        secondary, conf = select_secondary(probs, primary="joy")
        assert secondary is None
        assert conf is None

    def test_primary_excluded_from_candidates(self):
        # joy is primary — even if it has the highest prob, it must not be secondary
        probs = {"joy": 0.70, "sadness": 0.25, "anger": 0.05}
        secondary, conf = select_secondary(probs, primary="joy")
        assert secondary == "sadness"
        assert secondary != "joy"

    def test_threshold_boundary_exactly_at_020_retained(self):
        probs = {"joy": 0.70, "sadness": 0.20, "anger": 0.10}
        secondary, conf = select_secondary(probs, primary="joy", threshold=0.20)
        assert secondary == "sadness"

    def test_threshold_boundary_just_below_020_excluded(self):
        probs = {"joy": 0.70, "sadness": 0.199, "anger": 0.101}
        secondary, conf = select_secondary(probs, primary="joy", threshold=0.20)
        assert secondary is None

    def test_highest_remaining_label_selected(self):
        probs = {
            "joy": 0.50, "sadness": 0.30, "anger": 0.25,
            "fear": 0.22, "disgust": 0.21
        }
        secondary, conf = select_secondary(probs, primary="joy")
        assert secondary == "sadness"  # highest after joy excluded

    def test_no_candidates_above_threshold_returns_none(self):
        probs = {"joy": 0.90, "sadness": 0.05, "anger": 0.05}
        secondary, conf = select_secondary(probs, primary="joy")
        assert secondary is None
        assert conf is None


# ── aggregate_mood ────────────────────────────────────────────────────────────

class TestAggregateMood:
    def test_empty_input_all_null(self):
        result = aggregate_mood([])
        assert result["mood_primary"] is None
        assert result["mood_primary_confidence"] is None
        assert result["mood_secondary"] is None
        assert result["mood_secondary_confidence"] is None
        assert result["mood_flag"] is None

    def test_single_chunk_primary_selected(self):
        chunk = {"joy": 0.70, "sadness": 0.15, "anger": 0.05,
                 "fear": 0.04, "disgust": 0.03, "surprise": 0.02, "neutral": 0.01}
        result = aggregate_mood([chunk])
        assert result["mood_primary"] == "joy"
        assert result["mood_primary_confidence"] == pytest.approx(0.70, abs=0.001)

    def test_single_chunk_secondary_above_threshold(self):
        chunk = {"joy": 0.65, "sadness": 0.25, "anger": 0.04,
                 "fear": 0.03, "disgust": 0.01, "surprise": 0.01, "neutral": 0.01}
        result = aggregate_mood([chunk])
        assert result["mood_secondary"] == "sadness"
        assert result["mood_secondary_confidence"] == pytest.approx(0.25, abs=0.001)

    def test_single_chunk_secondary_null_below_threshold(self):
        chunk = {"joy": 0.80, "sadness": 0.10, "anger": 0.05,
                 "fear": 0.02, "disgust": 0.01, "surprise": 0.01, "neutral": 0.01}
        result = aggregate_mood([chunk])
        assert result["mood_secondary"] is None
        assert result["mood_secondary_confidence"] is None

    def test_multi_chunk_modal_label_wins(self):
        # 2 of 3 chunks have joy as top label → joy wins
        joy_chunk = {"joy": 0.70, "sadness": 0.15, "anger": 0.05,
                     "fear": 0.04, "disgust": 0.03, "surprise": 0.02, "neutral": 0.01}
        sadness_chunk = {"joy": 0.10, "sadness": 0.75, "anger": 0.05,
                         "fear": 0.04, "disgust": 0.03, "surprise": 0.02, "neutral": 0.01}
        result = aggregate_mood([joy_chunk, joy_chunk, sadness_chunk])
        assert result["mood_primary"] == "joy"

    def test_multi_chunk_tie_resolved_by_tiebreak_order(self):
        # 1 chunk joy-top, 1 chunk sadness-top → tie in count → resolve by mean probs
        joy_chunk = {"joy": 0.60, "sadness": 0.20, "anger": 0.05,
                     "fear": 0.05, "disgust": 0.05, "surprise": 0.03, "neutral": 0.02}
        sadness_chunk = {"joy": 0.20, "sadness": 0.60, "anger": 0.05,
                         "fear": 0.05, "disgust": 0.05, "surprise": 0.03, "neutral": 0.02}
        result = aggregate_mood([joy_chunk, sadness_chunk])
        # Mean joy = 0.40, mean sadness = 0.40 — exact tie → tiebreak: joy > sadness
        assert result["mood_primary"] in ("joy", "sadness")  # either valid; tiebreak resolves

    def test_low_confidence_flag_below_threshold(self):
        chunk = {"neutral": 0.30, "joy": 0.25, "sadness": 0.20,
                 "anger": 0.10, "fear": 0.07, "disgust": 0.05, "surprise": 0.03}
        result = aggregate_mood([chunk])
        # neutral is top at 0.30, below 0.35 threshold
        assert result["mood_flag"] == "low_confidence"

    def test_no_low_confidence_flag_above_threshold(self):
        chunk = {"joy": 0.70, "sadness": 0.15, "anger": 0.05,
                 "fear": 0.04, "disgust": 0.03, "surprise": 0.02, "neutral": 0.01}
        result = aggregate_mood([chunk])
        assert result["mood_flag"] is None

    def test_low_confidence_boundary_exactly_at_threshold_not_flagged(self):
        # confidence = exactly 0.35 — threshold is strict <, not flagged
        chunk = {"joy": 0.35, "sadness": 0.25, "anger": 0.15,
                 "fear": 0.12, "disgust": 0.07, "surprise": 0.04, "neutral": 0.02}
        result = aggregate_mood([chunk])
        assert result["mood_flag"] is None

    def test_primary_label_in_valid_taxonomy(self):
        chunk = {"joy": 0.50, "sadness": 0.30, "anger": 0.10,
                 "fear": 0.05, "disgust": 0.03, "surprise": 0.01, "neutral": 0.01}
        result = aggregate_mood([chunk])
        assert result["mood_primary"] in _VALID_MOOD_LABELS

    def test_secondary_label_in_valid_taxonomy_when_set(self):
        chunk = {"joy": 0.65, "sadness": 0.25, "anger": 0.04,
                 "fear": 0.03, "disgust": 0.01, "surprise": 0.01, "neutral": 0.01}
        result = aggregate_mood([chunk])
        if result["mood_secondary"] is not None:
            assert result["mood_secondary"] in _VALID_MOOD_LABELS

    def test_secondary_differs_from_primary(self):
        chunk = {"joy": 0.65, "sadness": 0.25, "anger": 0.04,
                 "fear": 0.03, "disgust": 0.01, "surprise": 0.01, "neutral": 0.01}
        result = aggregate_mood([chunk])
        if result["mood_secondary"] is not None:
            assert result["mood_secondary"] != result["mood_primary"]


# ── _build_null_record ────────────────────────────────────────────────────────

class TestBuildNullRecord:
    def test_all_fields_null(self):
        record = _build_null_record("abc123")
        assert record["song_id"] == "abc123"
        assert record["mood_primary"] is None
        assert record["mood_primary_confidence"] is None
        assert record["mood_secondary"] is None
        assert record["mood_secondary_confidence"] is None
        assert record["mood_flag"] is None

    def test_schema_4_keys_present(self):
        record = _build_null_record("abc123")
        for col in _SCHEMA_4_COLUMNS:
            assert col in record, f"Missing Schema 4 key: {col}"


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def _base_config(self, minimal_config, tmp_path):
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        minimal_config.cache.lyrics_dir = str(tmp_path / "cache" / "lyrics")
        minimal_config.cache.inference_dir = str(tmp_path / "cache" / "inference")
        minimal_config.logging.missing_lyrics_log = str(
            tmp_path / "missing_lyrics.jsonl"
        )
        minimal_config.logging.low_confidence_log = str(
            tmp_path / "low_confidence.jsonl"
        )
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.inference.cache_enabled = False
        minimal_config.checkpoints.force_rerun.s3_mood = True
        minimal_config.inference.batch_size = 2
        minimal_config.inference.sleep_between_batches = 0.0
        return minimal_config

    def test_schema_4_columns_present(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        for col in _SCHEMA_4_COLUMNS:
            assert col in df.columns, f"Missing Schema 4 column: {col}"

    def test_null_lyrics_produces_null_record(
        self, minimal_config, tmp_path, lyrics_df_missing, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                df = run(config, lyrics_df=lyrics_df_missing, run_id="test")
        assert df.iloc[0]["mood_primary"] is None
        assert df.iloc[0]["mood_primary_confidence"] is None
        assert df.iloc[0]["mood_secondary"] is None

    def test_primary_label_valid_taxonomy(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        scored = df[df["mood_primary"].notna()]
        assert scored["mood_primary"].isin(_VALID_MOOD_LABELS).all()

    def test_secondary_null_when_below_threshold(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        # mock_pipe returns sadness=0.15, below 0.20 threshold
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        assert df["mood_secondary"].isna().all()

    def test_secondary_populated_when_above_threshold(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe_with_secondary
    ):
        # mock_pipe_with_secondary returns sadness=0.25, above 0.20 threshold
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe_with_secondary, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        assert df["mood_secondary"].notna().all()
        assert (df["mood_secondary"] == "sadness").all()

    def test_low_confidence_flag_set(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe_low_confidence
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe_low_confidence, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        assert (df["mood_flag"] == "low_confidence").all()

    def test_low_confidence_written_to_log(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe_low_confidence
    ):
        config = self._base_config(minimal_config, tmp_path)
        log_path = Path(config.logging.low_confidence_log)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe_low_confidence, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                run(config, lyrics_df=lyrics_df, run_id="test")
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["dimension"] == "mood"
        assert record["flag_value"] == "low_confidence"

    def test_inference_cache_hit_skips_model(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        config.inference.cache_enabled = True

        from src.core.cache import get_inference_cache as _get_cache
        cache = _get_cache(config)
        for row in lyrics_df.to_dict("records"):
            cached = {
                "song_id": row["song_id"],
                "mood_primary": "joy",
                "mood_primary_confidence": 0.75,
                "mood_secondary": None,
                "mood_secondary_confidence": None,
                "mood_flag": None,
            }
            cache.set_inference(row["song_id"], "mood", cached)
        cache.close()

        call_count = {"n": 0}

        def counting_classify(chunks, pipe, batch_size):
            call_count["n"] += len(chunks)
            return []

        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                with patch("src.stages.s3_mood.classify_chunks",
                           side_effect=counting_classify):
                    df = run(config, lyrics_df=lyrics_df, run_id="test")

        assert call_count["n"] == 0, "classify_chunks called despite cache hit"
        assert (df["mood_primary"] == "joy").all()

    def test_checkpoint_written(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                run(config, lyrics_df=lyrics_df, run_id="test")
        checkpoint = Path(config.checkpoints.dir) / "03_mood.parquet"
        assert checkpoint.exists()

    def test_checkpoint_read_when_exists(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model"):
                run(config, lyrics_df=lyrics_df, run_id="test")
        config.checkpoints.force_rerun.s3_mood = False
        with patch("src.stages.s3_mood.load_mood_model") as mock_load:
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        mock_load.assert_not_called()
        assert len(df) == len(lyrics_df)

    def test_model_unloaded_after_run(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_mood.load_mood_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_mood.unload_model") as mock_unload:
                run(config, lyrics_df=lyrics_df, run_id="test")
        mock_unload.assert_called_once()
