"""
Tests for src/stages/s3_jungian.py

Covers:
  - api_unavailable path produces correct null record
  - retry fires on first failure, succeeds on second
  - parse_response failure sets api_unavailable not unhandled exception
  - evidence field populated for non-null primary
  - speculative flag written to low_confidence.jsonl
  - missing lyrics produces insufficient_evidence null record
  - cache hit skips API call
  - theme fallback result returned when request_theme=True
  - theme fallback None when Haiku returns no theme
  - call_haiku_for_theme_fallback returns partial Schema 5 dict
  - checkpoint read/write behavior
  - Schema 6 columns present in output
  - jungian_source always "haiku" for processed songs
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.stages.s3_jungian import (
    _build_null_record,
    _build_record_from_parsed,
    _process_song,
    call_haiku,
    call_haiku_for_theme_fallback,
    handle_api_failure,
    run,
)

# ── Schema 6 columns (locked P5) ─────────────────────────────────────────────
_SCHEMA_6_COLUMNS = [
    "song_id",
    "jungian_primary",
    "jungian_secondary",
    "jungian_confidence",
    "jungian_evidence",
    "jungian_flag",
    "jungian_source",
]

_VALID_ARCHETYPES = {
    "hero", "shadow", "anima_animus", "self",
    "trickster", "great_mother", "wise_old_man", "persona",
}

_VALID_THEMES = {
    "love_and_romance", "heartbreak_and_loss", "identity_and_self",
    "rebellion_and_defiance", "spirituality_and_faith",
    "materialism_and_ambition", "nostalgia_and_memory",
    "social_commentary", "hedonism_and_pleasure",
    "longing_and_desire", "conflict_and_struggle", "celebration_and_joy",
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
def valid_jungian_response():
    """Valid Haiku JSON response with Jungian primary and no theme."""
    return json.dumps({
        "jungian": {
            "primary": "anima_animus",
            "secondary": None,
            "confidence": "high",
            "evidence": ["wise men say only fools rush in"],
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


@pytest.fixture()
def valid_jungian_with_theme_response():
    """Valid Haiku JSON response with both Jungian and theme populated."""
    return json.dumps({
        "jungian": {
            "primary": "hero",
            "secondary": None,
            "confidence": "medium",
            "evidence": ["I built this world with my own two hands"],
            "flag": None,
        },
        "theme_fallback": {
            "requested": True,
            "primary": "identity_and_self",
            "primary_confidence": 0.75,
            "secondary": None,
            "secondary_confidence": 0.0,
        },
    })


@pytest.fixture()
def insufficient_evidence_response():
    """Valid Haiku JSON response where model found no archetype evidence."""
    return json.dumps({
        "jungian": {
            "primary": None,
            "secondary": None,
            "confidence": None,
            "evidence": [],
            "flag": "insufficient_evidence",
        },
        "theme_fallback": {
            "requested": False,
            "primary": None,
            "primary_confidence": 0.0,
            "secondary": None,
            "secondary_confidence": 0.0,
        },
    })


@pytest.fixture()
def speculative_response():
    """Haiku response flagged as speculative."""
    return json.dumps({
        "jungian": {
            "primary": "shadow",
            "secondary": None,
            "confidence": "low",
            "evidence": ["dark"],
            "flag": "speculative",
        },
        "theme_fallback": {
            "requested": False,
            "primary": None,
            "primary_confidence": 0.0,
            "secondary": None,
            "secondary_confidence": 0.0,
        },
    })


# ── _build_null_record ────────────────────────────────────────────────────────

class TestBuildNullRecord:
    def test_api_unavailable_flag_default(self):
        record = _build_null_record("abc123")
        assert record["jungian_flag"] == "api_unavailable"
        assert record["jungian_primary"] is None
        assert record["jungian_source"] is None

    def test_insufficient_evidence_flag(self):
        record = _build_null_record("abc123", flag="insufficient_evidence")
        assert record["jungian_flag"] == "insufficient_evidence"

    def test_schema_6_keys_present(self):
        record = _build_null_record("abc123")
        for col in _SCHEMA_6_COLUMNS:
            assert col in record, f"Missing Schema 6 key: {col}"


# ── handle_api_failure ────────────────────────────────────────────────────────

class TestHandleApiFailure:
    def test_returns_null_record_with_api_unavailable(self):
        record = handle_api_failure("abc123")
        assert record["jungian_flag"] == "api_unavailable"
        assert record["jungian_primary"] is None

    def test_schema_6_keys_present(self):
        record = handle_api_failure("abc123")
        for col in _SCHEMA_6_COLUMNS:
            assert col in record


# ── _build_record_from_parsed ─────────────────────────────────────────────────

class TestBuildRecordFromParsed:
    def test_fields_populated_correctly(self):
        parsed = {
            "jungian": {
                "primary": "hero",
                "secondary": None,
                "confidence": "high",
                "evidence": ["I rose from the ashes"],
                "flag": None,
            },
            "theme_fallback": {
                "requested": False,
                "primary": None,
                "primary_confidence": 0.0,
                "secondary": None,
                "secondary_confidence": 0.0,
            },
        }
        record = _build_record_from_parsed("abc123", parsed)
        assert record["jungian_primary"] == "hero"
        assert record["jungian_confidence"] == "high"
        assert record["jungian_evidence"] == ["I rose from the ashes"]
        assert record["jungian_flag"] is None

    def test_evidence_none_when_primary_null(self):
        parsed = {
            "jungian": {
                "primary": None,
                "secondary": None,
                "confidence": None,
                "evidence": None,
                "flag": "insufficient_evidence",
            },
            "theme_fallback": {
                "requested": False, "primary": None,
                "primary_confidence": 0.0, "secondary": None,
                "secondary_confidence": 0.0,
            },
        }
        record = _build_record_from_parsed("abc123", parsed)
        assert record["jungian_evidence"] is None
        assert record["jungian_primary"] is None


# ── call_haiku ────────────────────────────────────────────────────────────────

class TestCallHaiku:
    def _config(self, minimal_config):
        minimal_config.anthropic_api_key = "fake_key"
        minimal_config.jungian.haiku_model = "claude-haiku-3-20240307"
        minimal_config.jungian.max_retries = 2
        minimal_config.jungian.retry_sleep = 0.0
        return minimal_config

    def test_returns_text_on_success(self, minimal_config):
        config = self._config(minimal_config)
        mock_block = MagicMock()
        mock_block.text = '{"jungian": {}, "theme_fallback": {}}'
        mock_message = MagicMock()
        mock_message.content = [mock_block]

        with patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.return_value = mock_message
            result = call_haiku("test prompt", config)

        assert result == '{"jungian": {}, "theme_fallback": {}}'

    def test_retries_on_first_failure_succeeds_on_second(self, minimal_config):
        config = self._config(minimal_config)
        config.jungian.max_retries = 2

        mock_block = MagicMock()
        mock_block.text = '{"jungian": {}, "theme_fallback": {}}'
        mock_message = MagicMock()
        mock_message.content = [mock_block]

        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("Transient API error")
            return mock_message

        with patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.side_effect = side_effect
            result = call_haiku("test prompt", config)

        assert call_count["n"] == 2
        assert result is not None

    def test_returns_none_after_all_retries_fail(self, minimal_config):
        config = self._config(minimal_config)
        config.jungian.max_retries = 2

        with patch("anthropic.Anthropic") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.messages.create.side_effect = Exception("Persistent failure")
            result = call_haiku("test prompt", config)

        assert result is None


# ── _process_song ─────────────────────────────────────────────────────────────

class TestProcessSong:
    def _config(self, minimal_config):
        minimal_config.anthropic_api_key = "fake_key"
        minimal_config.jungian.haiku_model = "claude-haiku-3-20240307"
        minimal_config.jungian.max_retries = 1
        minimal_config.jungian.retry_sleep = 0.0
        return minimal_config

    def test_valid_response_produces_correct_record(
        self, minimal_config, valid_jungian_response
    ):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            record, theme = _process_song(
                song_id="abc123",
                lyrics="test lyrics",
                title="Test Song",
                artist="Test Artist",
                request_theme=False,
                config=config,
                run_id="test",
            )

        assert record["jungian_primary"] == "anima_animus"
        assert record["jungian_confidence"] == "high"
        assert record["jungian_evidence"] == ["wise men say only fools rush in"]
        assert record["jungian_flag"] is None
        assert record["jungian_source"] == "haiku"
        assert theme is None

    def test_api_none_returns_api_unavailable(self, minimal_config):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku", return_value=None):
            record, theme = _process_song(
                song_id="abc123",
                lyrics="test lyrics",
                title="Test Song",
                artist="Test Artist",
                request_theme=False,
                config=config,
                run_id="test",
            )
        assert record["jungian_flag"] == "api_unavailable"
        assert record["jungian_primary"] is None
        assert theme is None

    def test_parse_failure_returns_api_unavailable(self, minimal_config):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value="this is not json at all"):
            record, theme = _process_song(
                song_id="abc123",
                lyrics="test lyrics",
                title="Test Song",
                artist="Test Artist",
                request_theme=False,
                config=config,
                run_id="test",
            )
        assert record["jungian_flag"] == "api_unavailable"
        assert theme is None

    def test_theme_fallback_returned_when_requested(
        self, minimal_config, valid_jungian_with_theme_response
    ):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_with_theme_response):
            record, theme = _process_song(
                song_id="abc123",
                lyrics="test lyrics",
                title="Test Song",
                artist="Test Artist",
                request_theme=True,
                config=config,
                run_id="test",
            )

        assert theme is not None
        assert theme["theme_primary"] == "identity_and_self"
        assert theme["theme_source"] == "haiku"
        assert theme["theme_flag"] == "haiku_fallback"

    def test_theme_fallback_none_when_not_requested(
        self, minimal_config, valid_jungian_response
    ):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            _, theme = _process_song(
                song_id="abc123",
                lyrics="test",
                title="Test", artist="Artist",
                request_theme=False,
                config=config,
                run_id="test",
            )
        assert theme is None

    def test_insufficient_evidence_response_correct_record(
        self, minimal_config, insufficient_evidence_response
    ):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=insufficient_evidence_response):
            record, _ = _process_song(
                song_id="abc123",
                lyrics="test lyrics",
                title="Song", artist="Artist",
                request_theme=False,
                config=config,
                run_id="test",
            )
        assert record["jungian_primary"] is None
        assert record["jungian_flag"] == "insufficient_evidence"
        assert record["jungian_source"] == "haiku"


# ── call_haiku_for_theme_fallback ─────────────────────────────────────────────

class TestCallHaikuForThemeFallback:
    def _config(self, minimal_config):
        minimal_config.anthropic_api_key = "fake_key"
        minimal_config.jungian.haiku_model = "claude-haiku-3-20240307"
        minimal_config.jungian.max_retries = 1
        minimal_config.jungian.retry_sleep = 0.0
        return minimal_config

    def test_returns_partial_schema_5_dict(
        self, minimal_config, valid_jungian_with_theme_response
    ):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_with_theme_response):
            result = call_haiku_for_theme_fallback(
                song_id="abc123",
                lyrics="test lyrics",
                title="Test Song",
                artist="Test Artist",
                config=config,
                run_id="test",
            )

        assert result is not None
        assert result["theme_primary"] == "identity_and_self"
        assert result["theme_source"] == "haiku"
        assert result["theme_flag"] == "haiku_fallback"
        assert "song_id" in result

    def test_returns_none_when_no_theme_in_response(
        self, minimal_config, insufficient_evidence_response
    ):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=insufficient_evidence_response):
            result = call_haiku_for_theme_fallback(
                song_id="abc123",
                lyrics="test lyrics",
                title="Test Song",
                artist="Test Artist",
                config=config,
                run_id="test",
            )
        assert result is None

    def test_returns_none_on_api_failure(self, minimal_config):
        config = self._config(minimal_config)
        with patch("src.stages.s3_jungian.call_haiku", return_value=None):
            result = call_haiku_for_theme_fallback(
                song_id="abc123",
                lyrics="test",
                title="Song", artist="Artist",
                config=config,
                run_id="test",
            )
        assert result is None


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
        minimal_config.checkpoints.force_rerun.s3_jungian = True
        minimal_config.inference.batch_size = 2
        minimal_config.inference.sleep_between_batches = 0.0
        minimal_config.anthropic_api_key = "fake_key"
        minimal_config.jungian.haiku_model = "claude-haiku-3-20240307"
        minimal_config.jungian.max_retries = 1
        minimal_config.jungian.retry_sleep = 0.0
        return minimal_config

    def test_schema_6_columns_present(
        self, minimal_config, tmp_path, lyrics_df, valid_jungian_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        for col in _SCHEMA_6_COLUMNS:
            assert col in df.columns, f"Missing Schema 6 column: {col}"

    def test_missing_lyrics_produces_insufficient_evidence(
        self, minimal_config, tmp_path, lyrics_df_missing
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku") as mock_haiku:
            df = run(config, lyrics_df=lyrics_df_missing, run_id="test")
        mock_haiku.assert_not_called()
        assert df.iloc[0]["jungian_flag"] == "insufficient_evidence"
        assert df.iloc[0]["jungian_primary"] is None

    def test_api_unavailable_produces_null_record(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku", return_value=None):
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        assert (df["jungian_flag"] == "api_unavailable").all()
        assert df["jungian_primary"].isna().all()

    def test_valid_response_evidence_field_populated(
        self, minimal_config, tmp_path, lyrics_df, valid_jungian_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        for evidence in df["jungian_evidence"].dropna():
            assert isinstance(evidence, list)
            assert len(evidence) >= 1

    def test_speculative_flag_written_to_log(
        self, minimal_config, tmp_path, lyrics_df, speculative_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        log_path = Path(config.logging.low_confidence_log)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=speculative_response):
            run(config, lyrics_df=lyrics_df, run_id="test")
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["dimension"] == "jungian"
        assert record["flag_value"] == "speculative"

    def test_jungian_source_always_haiku_for_processed_songs(
        self, minimal_config, tmp_path, lyrics_df, valid_jungian_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        processed = df[df["jungian_flag"] != "insufficient_evidence"]
        assert (processed["jungian_source"] == "haiku").all()

    def test_inference_cache_hit_skips_api_call(
        self, minimal_config, tmp_path, lyrics_df, valid_jungian_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        config.inference.cache_enabled = True

        from src.core.cache import get_inference_cache as _get_cache
        cache = _get_cache(config)
        for row in lyrics_df.to_dict("records"):
            cached = {
                "song_id": row["song_id"],
                "jungian_primary": "hero",
                "jungian_secondary": None,
                "jungian_confidence": "high",
                "jungian_evidence": ["I rose above"],
                "jungian_flag": None,
                "jungian_source": "haiku",
            }
            cache.set_inference(row["song_id"], "jungian", cached)
        cache.close()

        with patch("src.stages.s3_jungian.call_haiku") as mock_haiku:
            df = run(config, lyrics_df=lyrics_df, run_id="test")

        mock_haiku.assert_not_called()
        assert (df["jungian_primary"] == "hero").all()

    def test_checkpoint_written(
        self, minimal_config, tmp_path, lyrics_df, valid_jungian_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            run(config, lyrics_df=lyrics_df, run_id="test")
        checkpoint = Path(config.checkpoints.dir) / "03_jungian.parquet"
        assert checkpoint.exists()

    def test_checkpoint_read_when_exists(
        self, minimal_config, tmp_path, lyrics_df, valid_jungian_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            run(config, lyrics_df=lyrics_df, run_id="test")
        config.checkpoints.force_rerun.s3_jungian = False
        with patch("src.stages.s3_jungian.call_haiku") as mock_haiku:
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        mock_haiku.assert_not_called()
        assert len(df) == len(lyrics_df)

    def test_primary_in_valid_archetype_set_when_not_null(
        self, minimal_config, tmp_path, lyrics_df, valid_jungian_response
    ):
        config = self._base_config(minimal_config, tmp_path)
        with patch("src.stages.s3_jungian.call_haiku",
                   return_value=valid_jungian_response):
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        non_null = df[df["jungian_primary"].notna()]
        assert non_null["jungian_primary"].isin(_VALID_ARCHETYPES).all()
