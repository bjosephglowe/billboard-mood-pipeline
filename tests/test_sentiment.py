"""
Tests for src/stages/s3_sentiment.py

Covers:
  - assign_bin: all five bins, boundary values, edge clamp
  - chunk_lyrics: empty input, single chunk, multi-chunk split
  - aggregate_scores: null path, single chunk, multi-chunk weighted mean
  - low_confidence flag at threshold boundary
  - score range enforcement (-1.0 to 1.0)
  - null lyrics produces all-null Schema 3 record
  - inference cache hit skips model call
  - checkpoint read/write behavior
  - Schema 3 columns present in output
"""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.stages.s3_sentiment import (
    _build_null_record,
    aggregate_scores,
    assign_bin,
    chunk_lyrics,
    score_chunks,
    run,
)

# ── Schema 3 columns (locked P5) ─────────────────────────────────────────────
_SCHEMA_3_COLUMNS = [
    "song_id",
    "sentiment_score",
    "sentiment_bin",
    "sentiment_confidence",
    "sentiment_flag",
    "sentiment_chunk_count",
]


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
    positive=0.80, neutral=0.12, negative=0.08
    Continuous score = 0.80*1.0 + 0.12*0.0 + 0.08*(-1.0) = 0.72
    0.72 >= 0.60 → bin = strongly_positive
    Confidence = 0.80
    """
    def _pipe(texts, truncation=True, max_length=512):
        return [
            [
                {"label": "positive", "score": 0.80},
                {"label": "neutral",  "score": 0.12},
                {"label": "negative", "score": 0.08},
            ]
            for _ in texts
        ]
    return _pipe


# ── assign_bin ────────────────────────────────────────────────────────────────

class TestAssignBin:
    # strongly_positive: score >= 0.60
    def test_strongly_positive_interior(self):
        assert assign_bin(0.75) == "strongly_positive"

    def test_strongly_positive_lower_bound(self):
        assert assign_bin(0.60) == "strongly_positive"

    def test_strongly_positive_at_one(self):
        assert assign_bin(1.00) == "strongly_positive"

    # positive: 0.20 <= score < 0.60
    def test_positive_interior(self):
        assert assign_bin(0.40) == "positive"

    def test_positive_lower_bound(self):
        assert assign_bin(0.20) == "positive"

    def test_positive_just_below_strongly_positive(self):
        assert assign_bin(0.599) == "positive"

    # neutral: -0.20 <= score < 0.20
    def test_neutral_at_zero(self):
        assert assign_bin(0.00) == "neutral"

    def test_neutral_positive_side(self):
        assert assign_bin(0.10) == "neutral"

    def test_neutral_negative_side(self):
        assert assign_bin(-0.10) == "neutral"

    def test_neutral_lower_bound(self):
        # -0.20 is the lower bound of neutral (inclusive)
        assert assign_bin(-0.20) == "neutral"

    def test_neutral_just_below_upper_bound(self):
        assert assign_bin(0.199) == "neutral"

    # negative: -0.60 <= score < -0.20
    def test_negative_interior(self):
        assert assign_bin(-0.30) == "negative"

    def test_negative_lower_bound(self):
        # -0.60 is the lower bound of negative (inclusive)
        assert assign_bin(-0.60) == "negative"

    def test_negative_just_above_neutral_boundary(self):
        # -0.201 is just inside the negative bin
        assert assign_bin(-0.201) == "negative"

    def test_negative_just_below_strongly_negative(self):
        assert assign_bin(-0.599) == "negative"

    # strongly_negative: score < -0.60
    def test_strongly_negative_interior(self):
        assert assign_bin(-0.70) == "strongly_negative"

    def test_strongly_negative_just_below_boundary(self):
        assert assign_bin(-0.601) == "strongly_negative"

    def test_strongly_negative_at_minus_one(self):
        assert assign_bin(-1.00) == "strongly_negative"

    # boundary precision
    def test_boundary_positive_to_strongly_positive(self):
        assert assign_bin(0.60) == "strongly_positive"
        assert assign_bin(0.599) == "positive"

    def test_boundary_neutral_upper(self):
        assert assign_bin(0.20) == "positive"
        assert assign_bin(0.199) == "neutral"

    def test_boundary_neutral_lower(self):
        # -0.20 is neutral (inclusive lower bound of neutral)
        assert assign_bin(-0.20) == "neutral"
        assert assign_bin(-0.201) == "negative"

    def test_boundary_strongly_negative(self):
        assert assign_bin(-0.60) == "negative"
        assert assign_bin(-0.601) == "strongly_negative"

    # clamp edge cases
    def test_clamp_above_1(self):
        assert assign_bin(1.01) == "strongly_positive"

    def test_clamp_below_minus_1(self):
        assert assign_bin(-1.01) == "strongly_negative"


# ── chunk_lyrics ──────────────────────────────────────────────────────────────

class TestChunkLyrics:
    def test_empty_string_returns_empty_list(self):
        assert chunk_lyrics("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_lyrics("   ") == []

    def test_short_lyrics_single_chunk(self):
        lyrics = "Line one\nLine two\nLine three\n"
        chunks = chunk_lyrics(lyrics, max_tokens=512)
        assert len(chunks) == 1
        assert "Line one" in chunks[0]

    def test_long_lyrics_splits_into_multiple_chunks(self):
        # max_tokens=50 → max_words ≈ 37
        long_line = " ".join(["word"] * 10)
        lyrics = "\n".join([long_line] * 10)  # 100 words total
        chunks = chunk_lyrics(lyrics, max_tokens=50)
        assert len(chunks) > 1

    def test_all_chunks_non_empty(self):
        long_line = " ".join(["word"] * 10)
        lyrics = "\n".join([long_line] * 10)
        chunks = chunk_lyrics(lyrics, max_tokens=50)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_no_words_lost_across_chunks(self):
        long_line = " ".join([f"word{i}" for i in range(20)])
        lyrics = "\n".join([long_line] * 5)
        chunks = chunk_lyrics(lyrics, max_tokens=50)
        combined_words = set(" ".join(chunks).split())
        original_words = set(lyrics.split())
        assert original_words == combined_words


# ── aggregate_scores ──────────────────────────────────────────────────────────

class TestAggregateScores:
    def test_empty_input_returns_null_record(self):
        result = aggregate_scores([], [])
        assert result["sentiment_score"] is None
        assert result["sentiment_bin"] is None
        assert result["sentiment_confidence"] is None
        assert result["sentiment_flag"] is None
        assert result["sentiment_chunk_count"] == 0

    def test_single_chunk_score_and_bin(self):
        # score = 0.80 - 0.08 = 0.72 → strongly_positive (>= 0.60)
        scores = [{"positive": 0.80, "neutral": 0.12, "negative": 0.08}]
        lengths = [100]
        result = aggregate_scores(scores, lengths)
        expected_score = 0.80 * 1.0 + 0.12 * 0.0 + 0.08 * -1.0
        assert abs(result["sentiment_score"] - expected_score) < 0.001
        assert result["sentiment_bin"] == "strongly_positive"
        assert result["sentiment_confidence"] == pytest.approx(0.80, abs=0.001)
        assert result["sentiment_flag"] is None
        assert result["sentiment_chunk_count"] == 1

    def test_single_chunk_positive_bin(self):
        # score = 0.50 - 0.10 = 0.40 → positive (0.20 <= score < 0.60)
        scores = [{"positive": 0.50, "neutral": 0.40, "negative": 0.10}]
        lengths = [100]
        result = aggregate_scores(scores, lengths)
        assert result["sentiment_bin"] == "positive"

    def test_single_chunk_negative(self):
        # score = 0.05 - 0.85 = -0.80 → strongly_negative
        scores = [{"positive": 0.05, "neutral": 0.10, "negative": 0.85}]
        lengths = [100]
        result = aggregate_scores(scores, lengths)
        assert result["sentiment_bin"] in ("negative", "strongly_negative")

    def test_weighted_mean_favours_longer_chunk(self):
        # Chunk 1: strongly positive (200 words)
        # Chunk 2: strongly negative (10 words)
        # Weighted result should remain positive
        scores = [
            {"positive": 0.90, "neutral": 0.05, "negative": 0.05},
            {"positive": 0.05, "neutral": 0.05, "negative": 0.90},
        ]
        lengths = [200, 10]
        result = aggregate_scores(scores, lengths)
        assert result["sentiment_score"] > 0

    def test_score_clamped_to_valid_range(self):
        scores = [{"positive": 1.0, "neutral": 0.0, "negative": 0.0}]
        lengths = [100]
        result = aggregate_scores(scores, lengths)
        assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_low_confidence_flag_below_threshold(self):
        # max prob = 0.40, below 0.45 threshold
        scores = [{"positive": 0.40, "neutral": 0.33, "negative": 0.27}]
        lengths = [100]
        result = aggregate_scores(scores, lengths)
        assert result["sentiment_flag"] == "low_confidence"

    def test_no_low_confidence_flag_above_threshold(self):
        # max prob = 0.80, above 0.45 threshold
        scores = [{"positive": 0.80, "neutral": 0.12, "negative": 0.08}]
        lengths = [100]
        result = aggregate_scores(scores, lengths)
        assert result["sentiment_flag"] is None

    def test_low_confidence_boundary_at_threshold_not_flagged(self):
        # confidence = exactly 0.45 — threshold is strict <, so 0.45 is NOT flagged
        scores = [{"positive": 0.45, "neutral": 0.30, "negative": 0.25}]
        lengths = [100]
        result = aggregate_scores(scores, lengths)
        assert result["sentiment_flag"] is None

    def test_chunk_count_correct(self):
        scores = [
            {"positive": 0.80, "neutral": 0.12, "negative": 0.08},
            {"positive": 0.70, "neutral": 0.20, "negative": 0.10},
            {"positive": 0.60, "neutral": 0.25, "negative": 0.15},
        ]
        lengths = [100, 100, 100]
        result = aggregate_scores(scores, lengths)
        assert result["sentiment_chunk_count"] == 3


# ── _build_null_record ────────────────────────────────────────────────────────

class TestBuildNullRecord:
    def test_all_analysis_fields_null(self):
        record = _build_null_record("abc123")
        assert record["song_id"] == "abc123"
        assert record["sentiment_score"] is None
        assert record["sentiment_bin"] is None
        assert record["sentiment_confidence"] is None
        assert record["sentiment_flag"] is None
        assert record["sentiment_chunk_count"] is None

    def test_schema_3_keys_present(self):
        record = _build_null_record("abc123")
        for col in _SCHEMA_3_COLUMNS:
            assert col in record, f"Missing Schema 3 key: {col}"


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
        minimal_config.checkpoints.force_rerun.s3_sentiment = True
        minimal_config.inference.batch_size = 2
        minimal_config.inference.sleep_between_batches = 0.0
        return minimal_config

    def test_schema_3_columns_present(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        for col in _SCHEMA_3_COLUMNS:
            assert col in df.columns, f"Missing Schema 3 column: {col}"

    def test_null_lyrics_produces_null_record(
        self, minimal_config, tmp_path, lyrics_df_missing, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                df = run(config, lyrics_df=lyrics_df_missing, run_id="test")
        assert df.iloc[0]["sentiment_score"] is None
        assert df.iloc[0]["sentiment_bin"] is None
        assert df.iloc[0]["sentiment_confidence"] is None
        assert df.iloc[0]["sentiment_chunk_count"] is None

    def test_score_in_valid_range(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        scored = df[df["sentiment_score"].notna()]
        assert (scored["sentiment_score"] >= -1.0).all()
        assert (scored["sentiment_score"] <= 1.0).all()

    def test_bin_assigned_for_all_scored_records(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        valid_bins = {
            "strongly_positive", "positive", "neutral",
            "negative", "strongly_negative",
        }
        scored = df[df["sentiment_bin"].notna()]
        assert scored["sentiment_bin"].isin(valid_bins).all()

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
                "sentiment_score": 0.55,
                "sentiment_bin": "positive",
                "sentiment_confidence": 0.88,
                "sentiment_flag": None,
                "sentiment_chunk_count": 1,
            }
            cache.set_inference(row["song_id"], "sentiment", cached)
        cache.close()

        call_count = {"n": 0}

        def counting_score(chunks, pipe, batch_size):
            call_count["n"] += len(chunks)
            return []

        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                with patch("src.stages.s3_sentiment.score_chunks",
                           side_effect=counting_score):
                    df = run(config, lyrics_df=lyrics_df, run_id="test")

        assert call_count["n"] == 0, "score_chunks called despite cache hit"
        assert (df["sentiment_score"] == 0.55).all()

    def test_checkpoint_written(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                run(config, lyrics_df=lyrics_df, run_id="test")
        checkpoint = Path(config.checkpoints.dir) / "03_sentiment.parquet"
        assert checkpoint.exists()

    def test_checkpoint_read_when_exists(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                run(config, lyrics_df=lyrics_df, run_id="test")

        config.checkpoints.force_rerun.s3_sentiment = False
        with patch("src.stages.s3_sentiment.load_sentiment_model") as mock_load:
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        mock_load.assert_not_called()
        assert len(df) == len(lyrics_df)

    def test_model_unloaded_after_run(
        self, minimal_config, tmp_path, lyrics_df, mock_pipe
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(mock_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model") as mock_unload:
                run(config, lyrics_df=lyrics_df, run_id="test")
        mock_unload.assert_called_once()

    def test_low_confidence_written_to_log(
        self, minimal_config, tmp_path
    ):
        config = self._base_config(minimal_config, tmp_path)

        row = {
            "song_id": "abc123def456abcd",
            "year": 1993, "rank": 1,
            "title": "Test Song", "artist": "Test Artist",
            "decade": "1990s",
            "title_normalized": "test song",
            "artist_normalized": "test artist",
            "collision_flag": False,
            "lyrics": "some lyrics here to analyse",
            "lyrics_status": "found",
            "lyrics_source": "genius",
            "lyrics_truncated": False,
            "lyrics_word_count": 5,
            "lyrics_fetched_at": "2024-01-01T00:00:00Z",
            "lyrics_cache_hit": False,
        }
        df_in = pd.DataFrame([row])

        # max prob = 0.40 → below 0.45 threshold → low_confidence
        def low_conf_pipe(texts, truncation=True, max_length=512):
            return [
                [
                    {"label": "positive", "score": 0.40},
                    {"label": "neutral",  "score": 0.33},
                    {"label": "negative", "score": 0.27},
                ]
                for _ in texts
            ]

        log_path = Path(config.logging.low_confidence_log)

        with patch("src.stages.s3_sentiment.load_sentiment_model",
                   return_value=(low_conf_pipe, "cpu")):
            with patch("src.stages.s3_sentiment.unload_model"):
                result = run(config, lyrics_df=df_in, run_id="test")

        assert result.iloc[0]["sentiment_flag"] == "low_confidence"
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip().split("\n")[0])
        assert record["dimension"] == "sentiment"
        assert record["flag_value"] == "low_confidence"
        assert record["song_id"] == "abc123def456abcd"
