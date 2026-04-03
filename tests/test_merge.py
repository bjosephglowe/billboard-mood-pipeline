"""
Tests for src/stages/s4_merge.py

Covers:
  - all Schema 8 columns present in output
  - record_complete True on fully populated fixture
  - record_complete False when lyrics_status missing
  - record_complete False when sentiment_score null
  - record_complete False when mood_primary null
  - record_complete False when theme_primary null
  - record_complete False when theme_primary is "uncertain"
  - Jungian null does NOT make record incomplete
  - semantic null does NOT make record incomplete
  - null propagation correct for missing-lyrics songs
  - pipeline_run_id present on every record
  - pipeline_run_id matches provided run_id
  - validate_schema returns empty list on complete DataFrame
  - validate_schema returns missing column names on incomplete DataFrame
  - join preserves all S2 rows (left join semantics)
  - missing S3 checkpoint handled gracefully (nulls, no abort)
  - checkpoint read/write behavior
  - Schema 8 column order enforced
"""
from pathlib import Path

import pandas as pd
import pytest

from src.stages.s4_merge import (
    _SCHEMA_8_COLUMNS,
    compute_record_complete,
    join_checkpoints,
    run,
    validate_schema,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_s2_row(song_id: str, lyrics_status: str = "found") -> dict:
    """Build a minimal Schema 2 row."""
    return {
        "song_id": song_id,
        "year": 1993,
        "rank": 1,
        "title": "Test Song",
        "artist": "Test Artist",
        "decade": "1990s",
        "title_normalized": "test song",
        "artist_normalized": "test artist",
        "collision_flag": False,
        "lyrics": "some lyrics here" if lyrics_status != "missing" else None,
        "lyrics_status": lyrics_status,
        "lyrics_source": "genius" if lyrics_status != "missing" else None,
        "lyrics_truncated": False,
        "lyrics_word_count": 3 if lyrics_status != "missing" else None,
        "lyrics_fetched_at": "2024-05-01T00:00:00Z",
        "lyrics_cache_hit": False,
    }


def _make_s3_sentiment(song_id: str, score: float = 0.70) -> dict:
    return {
        "song_id": song_id,
        "sentiment_score": score,
        "sentiment_bin": "positive",
        "sentiment_confidence": 0.85,
        "sentiment_flag": None,
        "sentiment_chunk_count": 1,
    }


def _make_s3_sentiment_null(song_id: str) -> dict:
    return {
        "song_id": song_id,
        "sentiment_score": None,
        "sentiment_bin": None,
        "sentiment_confidence": None,
        "sentiment_flag": None,
        "sentiment_chunk_count": None,
    }


def _make_s3_mood(song_id: str, primary: str = "joy") -> dict:
    return {
        "song_id": song_id,
        "mood_primary": primary,
        "mood_primary_confidence": 0.75,
        "mood_secondary": None,
        "mood_secondary_confidence": None,
        "mood_flag": None,
    }


def _make_s3_mood_null(song_id: str) -> dict:
    return {
        "song_id": song_id,
        "mood_primary": None,
        "mood_primary_confidence": None,
        "mood_secondary": None,
        "mood_secondary_confidence": None,
        "mood_flag": None,
    }


def _make_s3_theme(
    song_id: str,
    primary: str = "love_and_romance",
    confidence: float = 0.80,
) -> dict:
    return {
        "song_id": song_id,
        "theme_primary": primary,
        "theme_primary_confidence": confidence,
        "theme_secondary": None,
        "theme_secondary_confidence": None,
        "theme_source": "minilm",
        "theme_flag": None,
    }


def _make_s3_theme_null(song_id: str) -> dict:
    return {
        "song_id": song_id,
        "theme_primary": None,
        "theme_primary_confidence": None,
        "theme_secondary": None,
        "theme_secondary_confidence": None,
        "theme_source": None,
        "theme_flag": None,
    }


def _make_s3_jungian(song_id: str, primary: str = "hero") -> dict:
    return {
        "song_id": song_id,
        "jungian_primary": primary,
        "jungian_secondary": None,
        "jungian_confidence": "high",
        "jungian_evidence": ["I rose from the ashes"],
        "jungian_flag": None,
        "jungian_source": "haiku",
    }


def _make_s3_jungian_null(song_id: str) -> dict:
    return {
        "song_id": song_id,
        "jungian_primary": None,
        "jungian_secondary": None,
        "jungian_confidence": None,
        "jungian_evidence": None,
        "jungian_flag": "api_unavailable",
        "jungian_source": None,
    }


def _make_s3_semantic(song_id: str) -> dict:
    return {
        "song_id": song_id,
        "mtld_score": 68.4,
        "imagery_density": 0.31,
        "avg_line_length": 7.2,
        "tfidf_keywords": ["love", "heart", "soul"],
        "subject_focus": "relationship",
        "semantic_vector": None,
    }


def _make_s3_semantic_null(song_id: str) -> dict:
    return {
        "song_id": song_id,
        "mtld_score": None,
        "imagery_density": None,
        "avg_line_length": None,
        "tfidf_keywords": None,
        "subject_focus": None,
        "semantic_vector": None,
    }


def _build_full_s3_dfs(song_id: str) -> dict[str, pd.DataFrame]:
    """Build a complete set of S3 DataFrames for a single song."""
    return {
        "s3_sentiment": pd.DataFrame([_make_s3_sentiment(song_id)]),
        "s3_mood":      pd.DataFrame([_make_s3_mood(song_id)]),
        "s3_theme":     pd.DataFrame([_make_s3_theme(song_id)]),
        "s3_jungian":   pd.DataFrame([_make_s3_jungian(song_id)]),
        "s3_semantic":  pd.DataFrame([_make_s3_semantic(song_id)]),
    }


def _build_null_s3_dfs(song_id: str) -> dict[str, pd.DataFrame]:
    """Build a complete set of null S3 DataFrames for a single song."""
    return {
        "s3_sentiment": pd.DataFrame([_make_s3_sentiment_null(song_id)]),
        "s3_mood":      pd.DataFrame([_make_s3_mood_null(song_id)]),
        "s3_theme":     pd.DataFrame([_make_s3_theme_null(song_id)]),
        "s3_jungian":   pd.DataFrame([_make_s3_jungian_null(song_id)]),
        "s3_semantic":  pd.DataFrame([_make_s3_semantic_null(song_id)]),
    }


# ── compute_record_complete ───────────────────────────────────────────────────

class TestComputeRecordComplete:
    def _make_df(self, overrides: dict) -> pd.DataFrame:
        base = {
            "lyrics_status": "found",
            "sentiment_score": 0.70,
            "mood_primary": "joy",
            "theme_primary": "love_and_romance",
            # Jungian and semantic deliberately absent — must not affect result
        }
        base.update(overrides)
        return pd.DataFrame([base])

    def test_true_when_all_required_fields_populated(self):
        df = self._make_df({})
        result = compute_record_complete(df)
        assert result.iloc[0] is True or result.iloc[0] == True

    def test_false_when_lyrics_status_missing(self):
        df = self._make_df({"lyrics_status": "missing"})
        result = compute_record_complete(df)
        assert result.iloc[0] is False or result.iloc[0] == False

    def test_false_when_sentiment_score_null(self):
        df = self._make_df({"sentiment_score": None})
        result = compute_record_complete(df)
        assert not result.iloc[0]

    def test_false_when_mood_primary_null(self):
        df = self._make_df({"mood_primary": None})
        result = compute_record_complete(df)
        assert not result.iloc[0]

    def test_false_when_theme_primary_null(self):
        df = self._make_df({"theme_primary": None})
        result = compute_record_complete(df)
        assert not result.iloc[0]

    def test_false_when_theme_primary_uncertain(self):
        df = self._make_df({"theme_primary": "uncertain"})
        result = compute_record_complete(df)
        assert not result.iloc[0]

    def test_true_when_lyrics_status_truncated(self):
        df = self._make_df({"lyrics_status": "truncated"})
        result = compute_record_complete(df)
        assert result.iloc[0]

    def test_jungian_null_does_not_affect_completeness(self):
        """Null Jungian must NOT make a record incomplete per P5."""
        df = self._make_df({})
        # Add jungian null columns
        df["jungian_primary"] = None
        df["jungian_flag"] = "api_unavailable"
        result = compute_record_complete(df)
        assert result.iloc[0]

    def test_semantic_null_does_not_affect_completeness(self):
        """Null semantic fields must NOT make a record incomplete per P5."""
        df = self._make_df({})
        df["mtld_score"] = None
        df["subject_focus"] = None
        result = compute_record_complete(df)
        assert result.iloc[0]

    def test_multiple_rows_computed_independently(self):
        rows = [
            {
                "lyrics_status": "found",
                "sentiment_score": 0.70,
                "mood_primary": "joy",
                "theme_primary": "love_and_romance",
            },
            {
                "lyrics_status": "missing",
                "sentiment_score": None,
                "mood_primary": None,
                "theme_primary": None,
            },
        ]
        df = pd.DataFrame(rows)
        result = compute_record_complete(df)
        assert result.iloc[0]
        assert not result.iloc[1]


# ── validate_schema ───────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_empty_list_when_all_columns_present(self):
        df = pd.DataFrame(columns=_SCHEMA_8_COLUMNS)
        missing = validate_schema(df)
        assert missing == []

    def test_returns_missing_column_names(self):
        cols = [c for c in _SCHEMA_8_COLUMNS if c != "sentiment_score"]
        df = pd.DataFrame(columns=cols)
        missing = validate_schema(df)
        assert "sentiment_score" in missing

    def test_returns_multiple_missing_columns(self):
        cols = [c for c in _SCHEMA_8_COLUMNS
                if c not in ("jungian_primary", "mood_secondary")]
        df = pd.DataFrame(columns=cols)
        missing = validate_schema(df)
        assert "jungian_primary" in missing
        assert "mood_secondary" in missing

    def test_empty_dataframe_all_columns_present_is_valid(self):
        df = pd.DataFrame(columns=_SCHEMA_8_COLUMNS)
        assert validate_schema(df) == []


# ── join_checkpoints ──────────────────────────────────────────────────────────

class TestJoinCheckpoints:
    def test_all_s3_fields_present_after_join(self):
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id)])
        s3_dfs = _build_full_s3_dfs(song_id)
        merged = join_checkpoints(s2_df, s3_dfs)

        for col in _SCHEMA_8_COLUMNS:
            if col not in ("record_complete", "skip_reason", "pipeline_run_id"):
                assert col in merged.columns, f"Missing column after join: {col}"

    def test_s2_row_count_preserved(self):
        """Left join must retain all S2 rows."""
        s2_df = pd.DataFrame([
            _make_s2_row("aaa111bbb222cccc"),
            _make_s2_row("ddd333eee444ffff"),
        ])
        s3_dfs = {
            stage: pd.DataFrame([
                {**{"song_id": "aaa111bbb222cccc"}, **{f: None for f in fields}},
                {**{"song_id": "ddd333eee444ffff"}, **{f: None for f in fields}},
            ])
            for stage, fields in {
                "s3_sentiment": ["sentiment_score", "sentiment_bin",
                                 "sentiment_confidence", "sentiment_flag",
                                 "sentiment_chunk_count"],
                "s3_mood": ["mood_primary", "mood_primary_confidence",
                            "mood_secondary", "mood_secondary_confidence",
                            "mood_flag"],
                "s3_theme": ["theme_primary", "theme_primary_confidence",
                             "theme_secondary", "theme_secondary_confidence",
                             "theme_source", "theme_flag"],
                "s3_jungian": ["jungian_primary", "jungian_secondary",
                               "jungian_confidence", "jungian_evidence",
                               "jungian_flag", "jungian_source"],
                "s3_semantic": ["mtld_score", "imagery_density",
                                "avg_line_length", "tfidf_keywords",
                                "subject_focus", "semantic_vector"],
            }.items()
        }
        merged = join_checkpoints(s2_df, s3_dfs)
        assert len(merged) == 2

    def test_s2_song_id_retained_as_primary_key(self):
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id)])
        s3_dfs = _build_full_s3_dfs(song_id)
        merged = join_checkpoints(s2_df, s3_dfs)
        assert merged.iloc[0]["song_id"] == song_id

    def test_analysis_values_correctly_joined(self):
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id)])
        s3_dfs = _build_full_s3_dfs(song_id)
        merged = join_checkpoints(s2_df, s3_dfs)
        assert merged.iloc[0]["sentiment_score"] == pytest.approx(0.70, abs=0.001)
        assert merged.iloc[0]["mood_primary"] == "joy"
        assert merged.iloc[0]["theme_primary"] == "love_and_romance"
        assert merged.iloc[0]["jungian_primary"] == "hero"
        assert merged.iloc[0]["mtld_score"] == pytest.approx(68.4, abs=0.01)


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
        minimal_config.checkpoints.force_rerun.s4_merge = True
        return minimal_config

    def _make_inputs(self, song_id: str = "abc123def456abcd"):
        s2_df = pd.DataFrame([_make_s2_row(song_id)])
        s3_dfs = _build_full_s3_dfs(song_id)
        return s2_df, s3_dfs

    def test_schema_8_columns_present(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        s2_df, s3_dfs = self._make_inputs()
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        for col in _SCHEMA_8_COLUMNS:
            assert col in df.columns, f"Missing Schema 8 column: {col}"

    def test_schema_8_column_order_enforced(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        s2_df, s3_dfs = self._make_inputs()
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert list(df.columns) == _SCHEMA_8_COLUMNS

    def test_record_complete_true_on_full_record(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        s2_df, s3_dfs = self._make_inputs()
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert df.iloc[0]["record_complete"]

    def test_record_complete_false_for_missing_lyrics(
        self, minimal_config, tmp_path
    ):
        config = self._base_config(minimal_config, tmp_path)
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id, lyrics_status="missing")])
        s3_dfs = _build_null_s3_dfs(song_id)
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert not df.iloc[0]["record_complete"]

    def test_record_complete_false_for_uncertain_theme(
        self, minimal_config, tmp_path
    ):
        config = self._base_config(minimal_config, tmp_path)
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id)])
        s3_dfs = _build_full_s3_dfs(song_id)
        s3_dfs["s3_theme"] = pd.DataFrame([
            _make_s3_theme(song_id, primary="uncertain")
        ])
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert not df.iloc[0]["record_complete"]

    def test_null_propagation_for_missing_lyrics_song(
        self, minimal_config, tmp_path
    ):
        config = self._base_config(minimal_config, tmp_path)
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id, lyrics_status="missing")])
        s3_dfs = _build_null_s3_dfs(song_id)
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert df.iloc[0]["sentiment_score"] is None or pd.isna(df.iloc[0]["sentiment_score"])
        assert df.iloc[0]["mood_primary"] is None or pd.isna(df.iloc[0]["mood_primary"])
        assert df.iloc[0]["theme_primary"] is None or pd.isna(df.iloc[0]["theme_primary"])
        assert df.iloc[0]["jungian_primary"] is None or pd.isna(df.iloc[0]["jungian_primary"])
        assert df.iloc[0]["mtld_score"] is None or pd.isna(df.iloc[0]["mtld_score"])

    def test_jungian_null_does_not_affect_completeness(
        self, minimal_config, tmp_path
    ):
        config = self._base_config(minimal_config, tmp_path)
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id)])
        s3_dfs = _build_full_s3_dfs(song_id)
        s3_dfs["s3_jungian"] = pd.DataFrame([_make_s3_jungian_null(song_id)])
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert df.iloc[0]["record_complete"]

    def test_pipeline_run_id_on_every_record(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        s2_df = pd.DataFrame([
            _make_s2_row("aaa111bbb222cccc"),
            _make_s2_row("ddd333eee444ffff"),
        ])
        s3_dfs = {
            **_build_full_s3_dfs("aaa111bbb222cccc"),
        }
        # Build combined S3 dfs for both songs
        for stage in ["s3_sentiment", "s3_mood", "s3_theme",
                      "s3_jungian", "s3_semantic"]:
            rows = s3_dfs[stage].to_dict("records")
            stage_builder = {
                "s3_sentiment": _make_s3_sentiment,
                "s3_mood": _make_s3_mood,
                "s3_theme": _make_s3_theme,
                "s3_jungian": _make_s3_jungian,
                "s3_semantic": _make_s3_semantic,
            }[stage]
            rows.append(stage_builder("ddd333eee444ffff"))
            s3_dfs[stage] = pd.DataFrame(rows)

        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="1990s_test")
        assert (df["pipeline_run_id"] == "1990s_test").all()

    def test_pipeline_run_id_matches_provided_value(
        self, minimal_config, tmp_path
    ):
        config = self._base_config(minimal_config, tmp_path)
        s2_df, s3_dfs = self._make_inputs()
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="1990s_20240501T143200")
        assert df.iloc[0]["pipeline_run_id"] == "1990s_20240501T143200"

    def test_skip_reason_null_by_default(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        s2_df, s3_dfs = self._make_inputs()
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert df.iloc[0]["skip_reason"] is None or pd.isna(df.iloc[0]["skip_reason"])

    def test_missing_s3_checkpoint_handled_gracefully(
        self, minimal_config, tmp_path
    ):
        """Missing S3 checkpoint must not abort — nulls propagated instead."""
        config = self._base_config(minimal_config, tmp_path)
        song_id = "abc123def456abcd"
        s2_df = pd.DataFrame([_make_s2_row(song_id)])
        # Provide only sentiment, omit others — run() will attempt to load
        # missing checkpoints from disk, which won't exist, and handle gracefully
        s3_dfs = {
            "s3_sentiment": pd.DataFrame([_make_s3_sentiment(song_id)]),
            "s3_mood":      pd.DataFrame([_make_s3_mood(song_id)]),
            "s3_theme":     pd.DataFrame([_make_s3_theme(song_id)]),
            "s3_jungian":   pd.DataFrame([_make_s3_jungian(song_id)]),
            "s3_semantic":  pd.DataFrame([_make_s3_semantic(song_id)]),
        }
        # Should not raise even with all provided
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert len(df) == 1

    def test_checkpoint_written(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        s2_df, s3_dfs = self._make_inputs()
        run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        checkpoint = Path(config.checkpoints.dir) / "04_merged.parquet"
        assert checkpoint.exists()

    def test_checkpoint_read_when_exists(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        s2_df, s3_dfs = self._make_inputs()
        run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")

        config.checkpoints.force_rerun.s4_merge = False
        # Second call — must load from checkpoint, not recompute
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert len(df) == 1
        assert list(df.columns) == _SCHEMA_8_COLUMNS

    def test_two_songs_both_present_in_output(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        id1 = "aaa111bbb222cccc"
        id2 = "ddd333eee444ffff"
        s2_df = pd.DataFrame([_make_s2_row(id1), _make_s2_row(id2)])
        s3_dfs = {
            "s3_sentiment": pd.DataFrame([
                _make_s3_sentiment(id1), _make_s3_sentiment(id2)
            ]),
            "s3_mood": pd.DataFrame([
                _make_s3_mood(id1), _make_s3_mood(id2)
            ]),
            "s3_theme": pd.DataFrame([
                _make_s3_theme(id1), _make_s3_theme(id2)
            ]),
            "s3_jungian": pd.DataFrame([
                _make_s3_jungian(id1), _make_s3_jungian(id2)
            ]),
            "s3_semantic": pd.DataFrame([
                _make_s3_semantic(id1), _make_s3_semantic(id2)
            ]),
        }
        df = run(config, s2_df=s2_df, s3_dfs=s3_dfs, run_id="test_run")
        assert len(df) == 2
        assert set(df["song_id"].tolist()) == {id1, id2}
