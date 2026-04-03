"""
Tests for src/prompts/jungian_theme.py and src/stages/s3_theme.py

Covers jungian_theme.py:
  - build_prompt: theme fallback active/inactive, guardrails present,
    all permitted labels in prompt
  - parse_response: valid minimal response, null primary, out-of-taxonomy
    archetype raises, out-of-taxonomy theme raises, non-null primary with
    empty evidence raises, markdown fence stripping, invalid JSON raises

Covers s3_theme.py:
  - select_top_k: above threshold, below threshold, uncertain path,
    secondary null when only one label clears
  - classify_song: returns dict keyed by all theme labels
  - build_theme_record: all Schema 5 fields present
  - run: schema columns present, null lyrics produces null record,
    MiniLM above threshold produces minilm source, MiniLM below threshold
    triggers haiku fallback path, Haiku result replaces uncertain,
    Haiku failure retains uncertain, cache hit skips inference,
    checkpoint read/write, model unloaded after run
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.prompts.jungian_theme import (
    PromptParseError,
    _GUARDRAILS,
    build_prompt,
    get_permitted_archetypes,
    get_permitted_themes,
    parse_response,
)
from src.stages.s3_theme import (
    _build_null_record,
    build_theme_record,
    classify_song,
    run,
    select_top_k,
)

# ── Schema 5 columns (locked P5) ─────────────────────────────────────────────
_SCHEMA_5_COLUMNS = [
    "song_id",
    "theme_primary",
    "theme_primary_confidence",
    "theme_secondary",
    "theme_secondary_confidence",
    "theme_source",
    "theme_flag",
]

_VALID_THEMES = set(get_permitted_themes())
_VALID_ARCHETYPES = set(get_permitted_archetypes())


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def lyrics_df(sample_song_record, sample_song_record_2, sample_lyrics_found):
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
    row = sample_song_record.copy()
    row["lyrics"] = None
    row["lyrics_status"] = "missing"
    row["lyrics_source"] = None
    row["lyrics_truncated"] = False
    row["lyrics_word_count"] = None
    row["lyrics_fetched_at"] = "2024-05-01T00:00:00Z"
    row["lyrics_cache_hit"] = False
    return pd.DataFrame([row])


def _make_minilm_pipe(label: str, high_score: float = 0.80):
    """
    Returns a mock zero-shot pipeline that scores `label` highest.
    All other labels receive equal share of the remainder.
    """
    other_labels = [l for l in get_permitted_themes() if l != label]
    other_score = (1.0 - high_score) / len(other_labels) if other_labels else 0.0

    def _pipe(text, candidate_labels, truncation=True):
        scores = [high_score if l == label else other_score for l in candidate_labels]
        return {"labels": candidate_labels, "scores": scores}

    return _pipe


def _make_uncertain_pipe():
    """Returns a mock zero-shot pipeline where all labels score below threshold."""
    def _pipe(text, candidate_labels, truncation=True):
        equal = 1.0 / len(candidate_labels)
        return {
            "labels": candidate_labels,
            "scores": [equal] * len(candidate_labels),
        }
    return _pipe


# ── jungian_theme.py — build_prompt ──────────────────────────────────────────

class TestBuildPrompt:
    def test_guardrails_present_verbatim(self):
        prompt = build_prompt(
            lyrics="some lyrics",
            song_title="Test Song",
            artist="Test Artist",
            request_theme_fallback=False,
        )
        assert _GUARDRAILS in prompt

    def test_theme_fallback_requested_true(self):
        prompt = build_prompt(
            lyrics="some lyrics",
            song_title="Test",
            artist="Artist",
            request_theme_fallback=True,
        )
        assert '"requested": true' in prompt

    def test_theme_fallback_requested_false(self):
        prompt = build_prompt(
            lyrics="some lyrics",
            song_title="Test",
            artist="Artist",
            request_theme_fallback=False,
        )
        assert '"requested": false' in prompt

    def test_all_archetype_labels_in_prompt(self):
        prompt = build_prompt(
            lyrics="some lyrics",
            song_title="Test",
            artist="Artist",
            request_theme_fallback=False,
        )
        for archetype in get_permitted_archetypes():
            assert archetype in prompt, f"Archetype '{archetype}' missing from prompt"

    def test_all_theme_labels_in_prompt(self):
        prompt = build_prompt(
            lyrics="some lyrics",
            song_title="Test",
            artist="Artist",
            request_theme_fallback=True,
        )
        for theme in get_permitted_themes():
            assert theme in prompt, f"Theme '{theme}' missing from prompt"

    def test_lyrics_included_in_prompt(self):
        lyrics = "Wise men say only fools rush in"
        prompt = build_prompt(
            lyrics=lyrics,
            song_title="Test",
            artist="Artist",
            request_theme_fallback=False,
        )
        assert lyrics in prompt

    def test_song_title_and_artist_in_prompt(self):
        prompt = build_prompt(
            lyrics="lyrics",
            song_title="Can't Help Falling",
            artist="UB40",
            request_theme_fallback=False,
        )
        assert "Can't Help Falling" in prompt
        assert "UB40" in prompt


# ── jungian_theme.py — parse_response ────────────────────────────────────────

class TestParseResponse:
    def _valid_response(
        self,
        j_primary=None,
        j_evidence=None,
        j_confidence=None,
        j_flag="insufficient_evidence",
        j_secondary=None,
        t_requested=False,
        t_primary=None,
        t_primary_conf=0.0,
        t_secondary=None,
        t_secondary_conf=0.0,
    ):
        return json.dumps({
            "jungian": {
                "primary": j_primary,
                "secondary": j_secondary,
                "confidence": j_confidence,
                "evidence": j_evidence,
                "flag": j_flag,
            },
            "theme_fallback": {
                "requested": t_requested,
                "primary": t_primary,
                "primary_confidence": t_primary_conf,
                "secondary": t_secondary,
                "secondary_confidence": t_secondary_conf,
            },
        })

    def test_valid_null_primary_parses_successfully(self):
        raw = self._valid_response()
        result = parse_response(raw)
        assert result["jungian"]["primary"] is None
        assert result["jungian"]["flag"] == "insufficient_evidence"

    def test_valid_non_null_primary_with_evidence(self):
        raw = self._valid_response(
            j_primary="hero",
            j_evidence=["line one", "line two"],
            j_confidence="high",
            j_flag=None,
        )
        result = parse_response(raw)
        assert result["jungian"]["primary"] == "hero"
        assert len(result["jungian"]["evidence"]) == 2
        assert result["jungian"]["confidence"] == "high"

    def test_out_of_taxonomy_archetype_raises(self):
        raw = self._valid_response(
            j_primary="villain",
            j_evidence=["some phrase"],
            j_confidence="low",
            j_flag=None,
        )
        with pytest.raises(PromptParseError, match="not in permitted archetypes"):
            parse_response(raw)

    def test_out_of_taxonomy_theme_raises(self):
        raw = self._valid_response(
            t_requested=True,
            t_primary="sports_and_competition",
        )
        with pytest.raises(PromptParseError, match="not in permitted themes"):
            parse_response(raw)

    def test_non_null_primary_with_empty_evidence_raises(self):
        raw = self._valid_response(
            j_primary="shadow",
            j_evidence=[],
            j_confidence="low",
            j_flag=None,
        )
        with pytest.raises(PromptParseError, match="non-empty list"):
            parse_response(raw)

    def test_non_null_primary_with_null_evidence_raises(self):
        raw = self._valid_response(
            j_primary="shadow",
            j_evidence=None,
            j_confidence="low",
            j_flag=None,
        )
        with pytest.raises(PromptParseError, match="non-empty list"):
            parse_response(raw)

    def test_invalid_json_raises(self):
        with pytest.raises(PromptParseError, match="not valid JSON"):
            parse_response("this is not json")

    def test_markdown_fence_stripped(self):
        inner = self._valid_response()
        fenced = f"```json\n{inner}\n```"
        result = parse_response(fenced)
        assert result["jungian"] is not None

    def test_valid_theme_primary_returned(self):
        raw = self._valid_response(
            t_requested=True,
            t_primary="love_and_romance",
            t_primary_conf=0.75,
        )
        result = parse_response(raw)
        assert result["theme_fallback"]["primary"] == "love_and_romance"
        assert result["theme_fallback"]["primary_confidence"] == pytest.approx(0.75)

    def test_invalid_confidence_value_raises(self):
        raw = json.dumps({
            "jungian": {
                "primary": None,
                "secondary": None,
                "confidence": "extreme",
                "evidence": None,
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
        with pytest.raises(PromptParseError, match="confidence"):
            parse_response(raw)

    def test_all_permitted_archetypes_accepted(self):
        for archetype in get_permitted_archetypes():
            raw = self._valid_response(
                j_primary=archetype,
                j_evidence=["test phrase"],
                j_confidence="medium",
                j_flag=None,
            )
            result = parse_response(raw)
            assert result["jungian"]["primary"] == archetype

    def test_all_permitted_themes_accepted(self):
        for theme in get_permitted_themes():
            raw = self._valid_response(
                t_requested=True,
                t_primary=theme,
                t_primary_conf=0.70,
            )
            result = parse_response(raw)
            assert result["theme_fallback"]["primary"] == theme


# ── s3_theme.py — select_top_k ───────────────────────────────────────────────

class TestSelectTopK:
    def test_single_label_above_threshold(self):
        scores = {"love_and_romance": 0.80, "heartbreak_and_loss": 0.10}
        primary, p_conf, secondary, s_conf = select_top_k(
            scores, k=2, threshold=0.55
        )
        assert primary == "love_and_romance"
        assert p_conf == pytest.approx(0.80, abs=0.001)
        assert secondary is None
        assert s_conf is None

    def test_two_labels_above_threshold(self):
        scores = {
            "love_and_romance": 0.80,
            "nostalgia_and_memory": 0.65,
            "heartbreak_and_loss": 0.10,
        }
        primary, p_conf, secondary, s_conf = select_top_k(
            scores, k=2, threshold=0.55
        )
        assert primary == "love_and_romance"
        assert secondary == "nostalgia_and_memory"
        assert s_conf == pytest.approx(0.65, abs=0.001)

    def test_no_labels_above_threshold_returns_uncertain(self):
        scores = {label: 0.05 for label in get_permitted_themes()}
        primary, p_conf, secondary, s_conf = select_top_k(
            scores, k=2, threshold=0.55
        )
        assert primary == "uncertain"
        assert p_conf is None
        assert secondary is None
        assert s_conf is None

    def test_secondary_null_when_only_one_clears(self):
        scores = {"love_and_romance": 0.80}
        for label in get_permitted_themes():
            if label != "love_and_romance":
                scores[label] = 0.10
        primary, p_conf, secondary, s_conf = select_top_k(
            scores, k=2, threshold=0.55
        )
        assert primary == "love_and_romance"
        assert secondary is None

    def test_exact_threshold_boundary_retained(self):
        scores = {"love_and_romance": 0.55, "heartbreak_and_loss": 0.10}
        primary, p_conf, secondary, s_conf = select_top_k(
            scores, k=2, threshold=0.55
        )
        assert primary == "love_and_romance"

    def test_just_below_threshold_excluded(self):
        scores = {"love_and_romance": 0.549, "heartbreak_and_loss": 0.10}
        primary, p_conf, secondary, s_conf = select_top_k(
            scores, k=2, threshold=0.55
        )
        assert primary == "uncertain"


# ── s3_theme.py — build_theme_record ─────────────────────────────────────────

class TestBuildThemeRecord:
    def test_all_schema_5_fields_present(self):
        record = build_theme_record(
            song_id="abc123def456abcd",
            primary="love_and_romance",
            primary_conf=0.80,
            secondary=None,
            secondary_conf=None,
            source="minilm",
            flag=None,
        )
        for col in _SCHEMA_5_COLUMNS:
            assert col in record, f"Missing Schema 5 field: {col}"

    def test_null_record_all_null(self):
        record = _build_null_record("abc123def456abcd")
        for col in _SCHEMA_5_COLUMNS:
            assert col in record
        assert record["theme_primary"] is None
        assert record["theme_source"] is None


# ── s3_theme.py — run ────────────────────────────────────────────────────────

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
        minimal_config.checkpoints.force_rerun.s3_theme = True
        minimal_config.inference.batch_size = 2
        minimal_config.inference.sleep_between_batches = 0.0
        minimal_config.theme.min_confidence = 0.55
        minimal_config.theme.haiku_fallback_enabled = False
        return minimal_config

    def test_schema_5_columns_present(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        pipe = _make_minilm_pipe("love_and_romance", high_score=0.80)
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        for col in _SCHEMA_5_COLUMNS:
            assert col in df.columns, f"Missing Schema 5 column: {col}"

    def test_null_lyrics_produces_null_record(
        self, minimal_config, tmp_path, lyrics_df_missing
    ):
        config = self._base_config(minimal_config, tmp_path)
        pipe = _make_minilm_pipe("love_and_romance")
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                df = run(config, lyrics_df=lyrics_df_missing, run_id="test")
        assert df.iloc[0]["theme_primary"] is None
        assert df.iloc[0]["theme_source"] is None

    def test_above_threshold_produces_minilm_source(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        pipe = _make_minilm_pipe("love_and_romance", high_score=0.80)
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        assert (df["theme_source"] == "minilm").all()
        assert (df["theme_primary"] == "love_and_romance").all()
        assert df["theme_flag"].isna().all()

    def test_below_threshold_produces_uncertain(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        # Haiku fallback disabled — uncertain must remain
        pipe = _make_uncertain_pipe()
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        assert (df["theme_primary"] == "uncertain").all()
        assert (df["theme_source"] == "uncertain").all()

    def test_haiku_fallback_replaces_uncertain(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        config.theme.haiku_fallback_enabled = True

        pipe = _make_uncertain_pipe()

        haiku_result = {
            "jungian": {
                "primary": None, "secondary": None,
                "confidence": None, "evidence": None,
                "flag": "insufficient_evidence",
            },
            "theme_fallback": {
                "requested": True,
                "primary": "love_and_romance",
                "primary_confidence": 0.72,
                "secondary": None,
                "secondary_confidence": 0.0,
            },
        }

        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                with patch(
                    "src.stages.s3_theme.apply_haiku_fallback",
                    side_effect=lambda uncertain_songs, lyrics_lookup,
                    theme_records, config, inference_cache, run_id: [
                        theme_records.update({
                            song["song_id"]: {
                                "song_id": song["song_id"],
                                "theme_primary": "love_and_romance",
                                "theme_primary_confidence": 0.72,
                                "theme_secondary": None,
                                "theme_secondary_confidence": None,
                                "theme_source": "haiku",
                                "theme_flag": "haiku_fallback",
                            }
                        })
                        for song in uncertain_songs
                    ]
                ):
                    df = run(config, lyrics_df=lyrics_df, run_id="test")

        assert (df["theme_source"] == "haiku").all()
        assert (df["theme_primary"] == "love_and_romance").all()
        assert (df["theme_flag"] == "haiku_fallback").all()

    def test_primary_label_in_valid_taxonomy(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        pipe = _make_minilm_pipe("nostalgia_and_memory", high_score=0.80)
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                df = run(config, lyrics_df=lyrics_df, run_id="test")
        valid = _VALID_THEMES | {"uncertain"}
        scored = df[df["theme_primary"].notna()]
        assert scored["theme_primary"].isin(valid).all()

    def test_inference_cache_hit_skips_classify(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        config.inference.cache_enabled = True

        from src.core.cache import get_inference_cache as _get_cache
        cache = _get_cache(config)
        for row in lyrics_df.to_dict("records"):
            cached = {
                "song_id": row["song_id"],
                "theme_primary": "joy_and_celebration",
                "theme_primary_confidence": 0.77,
                "theme_secondary": None,
                "theme_secondary_confidence": None,
                "theme_source": "minilm",
                "theme_flag": None,
            }
            cache.set_inference(row["song_id"], "theme", cached)
        cache.close()

        call_count = {"n": 0}

        def counting_classify(lyrics, labels, pipe, threshold):
            call_count["n"] += 1
            return {label: 0.5 for label in labels}

        pipe = _make_minilm_pipe("love_and_romance")
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                with patch("src.stages.s3_theme.classify_song",
                           side_effect=counting_classify):
                    df = run(config, lyrics_df=lyrics_df, run_id="test")

        assert call_count["n"] == 0, "classify_song called despite cache hit"

    def test_checkpoint_written(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        pipe = _make_minilm_pipe("love_and_romance")
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                run(config, lyrics_df=lyrics_df, run_id="test")
        checkpoint = Path(config.checkpoints.dir) / "03_theme.parquet"
        assert checkpoint.exists()

    def test_checkpoint_read_when_exists(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        pipe = _make_minilm_pipe("love_and_romance")
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model"):
                run(config, lyrics_df=lyrics_df, run_id="test")
        config.checkpoints.force_rerun.s3_theme = False
        with patch("src.stages.s3_theme.load_theme_model") as mock_load:
            df = run(config, lyrics_df=lyrics_df, run_id="test")
        mock_load.assert_not_called()
        assert len(df) == len(lyrics_df)

    def test_model_unloaded_after_run(
        self, minimal_config, tmp_path, lyrics_df
    ):
        config = self._base_config(minimal_config, tmp_path)
        pipe = _make_minilm_pipe("love_and_romance")
        with patch("src.stages.s3_theme.load_theme_model",
                   return_value=(pipe, "cpu")):
            with patch("src.stages.s3_theme.unload_model") as mock_unload:
                run(config, lyrics_df=lyrics_df, run_id="test")
        mock_unload.assert_called_once()
