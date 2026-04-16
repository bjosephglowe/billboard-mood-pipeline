"""
Tests for src/stages/s5_report.py

Covers:
  - compute_report_inputs produces dict with all Schema 11 keys
  - coverage metrics computed correctly against known fixtures
  - evaluate_gate returns True when all thresholds met
  - evaluate_gate returns False when any threshold missed
  - evaluate_gate fails on low lyrics coverage
  - evaluate_gate fails on high low-confidence theme rate
  - plot functions write PNG files to correct paths
  - render_validation_report writes markdown with all metric values
  - render_validation_report contains gate_pass result
  - write_ndjson produces valid JSON per line
  - write_ndjson one line per record
  - NDJSON output contains all Schema 8 fields
  - run() writes all four PNG files
  - run() writes validation_report.md
  - run() writes analysis NDJSON
  - run() returns dict with gate_pass key
  - checkpoint (report file) read path skips regeneration
"""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.stages.s5_report import (
    compute_report_inputs,
    evaluate_gate,
    plot_jungian_distribution,
    plot_mood_heatmap,
    plot_sentiment_drift,
    plot_theme_frequency,
    render_validation_report,
    run,
    write_ndjson,
)

# ── Schema 11 required keys (locked P5) ──────────────────────────────────────
_SCHEMA_11_KEYS = [
    "run_id",
    "decade",
    "year_range",
    "total_songs",
    "lyrics_found_count",
    "lyrics_truncated_count",
    "lyrics_missing_count",
    "lyrics_coverage_pct",
    "sentiment_coverage_pct",
    "theme_coverage_pct",
    "jungian_coverage_pct",
    "semantic_coverage_pct",
    "low_confidence_theme_rate",
    "record_complete_count",
    "record_complete_pct",
    "skipped_count",
    "theme_distribution",
    "mood_distribution",
    "jungian_distribution",
    "sentiment_mean",
    "sentiment_std",
    "model_versions",
    "pipeline_run_timestamp",
    "gate_pass",
]


# ── Fixture builders ──────────────────────────────────────────────────────────

def _make_merged_row(
    song_id: str,
    year: int = 1993,
    lyrics_status: str = "found",
    sentiment_score: float = 0.60,
    mood_primary: str = "joy",
    theme_primary: str = "love_and_romance",
    jungian_primary: str = "hero",
    mtld_score: float = 68.0,
    imagery_density: float = 0.30,
    avg_line_length: float = 7.0,
    tfidf_keywords: list = None,
    subject_focus: str = "relationship",
    record_complete: bool = True,
) -> dict:
    return {
        "song_id": song_id,
        "year": year,
        "rank": 1,
        "title": "Test Song",
        "artist": "Test Artist",
        "decade": "1990s",
        "title_normalized": "test song",
        "artist_normalized": "test artist",
        "collision_flag": False,
        "lyrics": "some lyrics" if lyrics_status != "missing" else None,
        "lyrics_status": lyrics_status,
        "lyrics_source": "genius" if lyrics_status != "missing" else None,
        "lyrics_truncated": False,
        "lyrics_word_count": 2 if lyrics_status != "missing" else None,
        "lyrics_fetched_at": "2024-05-01T00:00:00Z",
        "lyrics_cache_hit": False,
        "sentiment_score": sentiment_score,
        "sentiment_bin": "positive" if sentiment_score and sentiment_score > 0 else None,
        "sentiment_confidence": 0.85,
        "sentiment_flag": None,
        "sentiment_chunk_count": 1,
        "mood_primary": mood_primary,
        "mood_primary_confidence": 0.75,
        "mood_secondary": None,
        "mood_secondary_confidence": None,
        "mood_flag": None,
        "theme_primary": theme_primary,
        "theme_primary_confidence": 0.80,
        "theme_secondary": None,
        "theme_secondary_confidence": None,
        "theme_source": "minilm",
        "theme_flag": None,
        "jungian_primary": jungian_primary,
        "jungian_secondary": None,
        "jungian_confidence": "high",
        "jungian_evidence": ["a lyric phrase"],
        "jungian_flag": None,
        "jungian_source": "haiku",
        "mtld_score": mtld_score,
        "imagery_density": imagery_density,
        "avg_line_length": avg_line_length,
        "tfidf_keywords": tfidf_keywords or ["love", "heart"],
        "subject_focus": subject_focus,
        "semantic_vector": None,
        "record_complete": record_complete,
        "skip_reason": None,
        "pipeline_run_id": "1990s_test",
    }


@pytest.fixture()
def merged_df_full():
    """10-song fully-populated merged DataFrame — all thresholds met."""
    rows = [_make_merged_row(f"song{i:016d}") for i in range(10)]
    return pd.DataFrame(rows)


@pytest.fixture()
def merged_df_low_coverage():
    """6 found + 4 missing lyrics — lyrics coverage = 0.60, below 0.85."""
    rows = (
        [_make_merged_row(f"song{i:016d}") for i in range(6)]
        + [
            _make_merged_row(
                f"song{i:016d}",
                lyrics_status="missing",
                sentiment_score=None,
                mood_primary=None,
                theme_primary=None,
                jungian_primary=None,
                mtld_score=None,
                imagery_density=None,
                avg_line_length=None,
                tfidf_keywords=None,
                subject_focus=None,
                record_complete=False,
            )
            for i in range(6, 10)
        ]
    )
    return pd.DataFrame(rows)


@pytest.fixture()
def merged_df_uncertain_theme():
    """All songs have uncertain theme — theme coverage = 0.0."""
    rows = [
        _make_merged_row(
            f"song{i:016d}",
            theme_primary="uncertain",
            record_complete=False,
        )
        for i in range(10)
    ]
    return pd.DataFrame(rows)


# ── compute_report_inputs ─────────────────────────────────────────────────────

class TestComputeReportInputs:
    def test_all_schema_11_keys_present(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        for key in _SCHEMA_11_KEYS:
            assert key in inputs, f"Missing Schema 11 key: {key}"

    def test_total_songs_correct(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        assert inputs["total_songs"] == 10

    def test_lyrics_coverage_full(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        assert inputs["lyrics_coverage_pct"] == pytest.approx(1.0, abs=0.001)

    def test_lyrics_coverage_partial(self, minimal_config, merged_df_low_coverage):
        inputs = compute_report_inputs(
            merged_df_low_coverage, minimal_config, "test_run"
        )
        assert inputs["lyrics_coverage_pct"] == pytest.approx(0.60, abs=0.001)

    def test_lyrics_missing_count(self, minimal_config, merged_df_low_coverage):
        inputs = compute_report_inputs(
            merged_df_low_coverage, minimal_config, "test_run"
        )
        assert inputs["lyrics_missing_count"] == 4

    def test_theme_coverage_zero_when_all_uncertain(
        self, minimal_config, merged_df_uncertain_theme
    ):
        inputs = compute_report_inputs(
            merged_df_uncertain_theme, minimal_config, "test_run"
        )
        assert inputs["theme_coverage_pct"] == pytest.approx(0.0, abs=0.001)

    def test_theme_distribution_keys_present(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        assert "love_and_romance" in inputs["theme_distribution"]
        assert "uncertain" in inputs["theme_distribution"]

    def test_mood_distribution_keys_present(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        for label in ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]:
            assert label in inputs["mood_distribution"]

    def test_jungian_distribution_keys_present(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        for label in ["hero", "shadow", "anima_animus", "self",
                      "trickster", "great_mother", "wise_old_man", "persona"]:
            assert label in inputs["jungian_distribution"]

    def test_sentiment_mean_computed(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        assert inputs["sentiment_mean"] is not None
        assert isinstance(inputs["sentiment_mean"], float)

    def test_run_id_preserved(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "1990s_run")
        assert inputs["run_id"] == "1990s_run"

    def test_record_complete_count(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        assert inputs["record_complete_count"] == 10

    def test_model_versions_all_tasks_present(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        for task in ["sentiment", "mood", "theme", "jungian", "semantic"]:
            assert task in inputs["model_versions"]

    def test_semantic_coverage_full(self, minimal_config, merged_df_full):
        inputs = compute_report_inputs(merged_df_full, minimal_config, "test_run")
        assert inputs["semantic_coverage_pct"] == pytest.approx(1.0, abs=0.001)

    def test_low_conf_theme_rate_zero_when_all_haiku_fallback(self, minimal_config):
        """
        Songs classified via Haiku fallback (theme_flag=haiku_fallback) must
        NOT count toward low_confidence_theme_rate. Only uncertain counts.
        This is the core Issue 1 / Option B assertion.
        """
        rows = [
            _make_merged_row(
                f"song{i:016d}",
                theme_primary="love_and_romance",  # valid label, not uncertain
            )
            for i in range(10)
        ]
        # Set all theme_flag to haiku_fallback — old metric would score 1.0
        for row in rows:
            row["theme_flag"] = "haiku_fallback"
            row["theme_source"] = "haiku"
        df = pd.DataFrame(rows)
        inputs = compute_report_inputs(df, minimal_config, "test_run")
        assert inputs["low_confidence_theme_rate"] == pytest.approx(0.0, abs=0.001)

    def test_low_conf_theme_rate_counts_only_uncertain(self, minimal_config):
        """
        3 uncertain + 7 haiku-classified → rate = 0.30 (3/10).
        Haiku-classified songs do not count regardless of theme_flag.
        """
        rows = []
        for i in range(7):
            row = _make_merged_row(f"song{i:016d}", theme_primary="love_and_romance")
            row["theme_flag"] = "haiku_fallback"
            row["theme_source"] = "haiku"
            rows.append(row)
        for i in range(7, 10):
            row = _make_merged_row(f"song{i:016d}", theme_primary="uncertain",
                                   record_complete=False)
            row["theme_flag"] = None
            row["theme_source"] = "uncertain"
            rows.append(row)
        df = pd.DataFrame(rows)
        inputs = compute_report_inputs(df, minimal_config, "test_run")
        assert inputs["low_confidence_theme_rate"] == pytest.approx(0.30, abs=0.001)


# ── evaluate_gate ─────────────────────────────────────────────────────────────

class TestEvaluateGate:
    def _passing_inputs(self):
        return {
            "lyrics_coverage_pct":       0.90,
            "sentiment_coverage_pct":    0.90,
            "theme_coverage_pct":        0.80,
            "jungian_coverage_pct":      0.65,
            "semantic_coverage_pct":     0.90,
            "low_confidence_theme_rate": 0.10,
        }

    def test_returns_true_when_all_pass(self, minimal_config):
        result = evaluate_gate(self._passing_inputs(), minimal_config)
        assert result is True

    def test_returns_false_when_lyrics_coverage_low(self, minimal_config):
        inputs = self._passing_inputs()
        inputs["lyrics_coverage_pct"] = 0.70  # below 0.85
        result = evaluate_gate(inputs, minimal_config)
        assert result is False

    def test_returns_false_when_sentiment_coverage_low(self, minimal_config):
        inputs = self._passing_inputs()
        inputs["sentiment_coverage_pct"] = 0.80  # below 0.85
        result = evaluate_gate(inputs, minimal_config)
        assert result is False

    def test_returns_false_when_theme_coverage_low(self, minimal_config):
        inputs = self._passing_inputs()
        inputs["theme_coverage_pct"] = 0.70  # below 0.75
        result = evaluate_gate(inputs, minimal_config)
        assert result is False

    def test_returns_false_when_jungian_coverage_low(self, minimal_config):
        inputs = self._passing_inputs()
        inputs["jungian_coverage_pct"] = 0.50  # below 0.60
        result = evaluate_gate(inputs, minimal_config)
        assert result is False

    def test_returns_false_when_low_conf_theme_rate_high(self, minimal_config):
        inputs = self._passing_inputs()
        inputs["low_confidence_theme_rate"] = 0.30  # above 0.25
        result = evaluate_gate(inputs, minimal_config)
        assert result is False

    def test_returns_false_when_semantic_coverage_low(self, minimal_config):
        inputs = self._passing_inputs()
        inputs["semantic_coverage_pct"] = 0.80  # below 0.85
        result = evaluate_gate(inputs, minimal_config)
        assert result is False

    def test_boundary_exactly_at_threshold_passes(self, minimal_config):
        inputs = {
            "lyrics_coverage_pct":       0.85,   # exactly at threshold
            "sentiment_coverage_pct":    0.85,
            "theme_coverage_pct":        0.75,
            "jungian_coverage_pct":      0.60,
            "semantic_coverage_pct":     0.85,
            "low_confidence_theme_rate": 0.25,   # exactly at max threshold
        }
        result = evaluate_gate(inputs, minimal_config)
        assert result is True


# ── plot functions ────────────────────────────────────────────────────────────

class TestPlotFunctions:
    def test_sentiment_drift_writes_file(self, tmp_path, merged_df_full):
        path = str(tmp_path / "sentiment_drift.png")
        plot_sentiment_drift(merged_df_full, path)
        assert Path(path).exists()

    def test_mood_heatmap_writes_file(self, tmp_path, merged_df_full):
        path = str(tmp_path / "mood_heatmap.png")
        plot_mood_heatmap(merged_df_full, path)
        assert Path(path).exists()

    def test_theme_frequency_writes_file(self, tmp_path, merged_df_full):
        path = str(tmp_path / "theme_frequency.png")
        plot_theme_frequency(merged_df_full, path)
        assert Path(path).exists()

    def test_jungian_distribution_writes_file(self, tmp_path, merged_df_full):
        path = str(tmp_path / "jungian_distribution.png")
        plot_jungian_distribution(merged_df_full, path)
        assert Path(path).exists()

    def test_sentiment_drift_no_crash_on_empty(self, tmp_path):
        df = pd.DataFrame(columns=["year", "sentiment_score"])
        path = str(tmp_path / "sentiment_drift.png")
        plot_sentiment_drift(df, path)
        # Should log warning and return without writing or crashing

    def test_creates_parent_directory(self, tmp_path, merged_df_full):
        nested = str(tmp_path / "nested" / "dir" / "sentiment_drift.png")
        plot_sentiment_drift(merged_df_full, nested)
        assert Path(nested).exists()


# ── render_validation_report ──────────────────────────────────────────────────

class TestRenderValidationReport:
    def _make_inputs(self, gate_pass: bool = True) -> dict:
        return {
            "run_id": "1990s_test",
            "decade": "1990s",
            "year_range": [1990, 1999],
            "total_songs": 100,
            "lyrics_found_count": 90,
            "lyrics_truncated_count": 5,
            "lyrics_missing_count": 5,
            "lyrics_coverage_pct": 0.95,
            "sentiment_coverage_pct": 0.95,
            "theme_coverage_pct": 0.88,
            "jungian_coverage_pct": 0.72,
            "semantic_coverage_pct": 0.92,
            "low_confidence_theme_rate": 0.08,
            "record_complete_count": 85,
            "record_complete_pct": 0.85,
            "skipped_count": 0,
            "theme_distribution": {"love_and_romance": 20, "uncertain": 5},
            "mood_distribution": {"joy": 30, "sadness": 20},
            "jungian_distribution": {"hero": 15, "shadow": 10},
            "sentiment_mean": 0.42,
            "sentiment_std": 0.31,
            "model_versions": {
                "sentiment": "cardiffnlp/twitter-roberta",
                "mood": "j-hartmann/emotion-distilroberta",
                "theme": "cross-encoder/nli-MiniLM2-L6-H768",
                "jungian": "claude-haiku-3-20240307",
                "semantic": "spacy:en_core_web_sm",
            },
            "pipeline_run_timestamp": "1990s_test",
            "gate_pass": gate_pass,
        }

    def test_writes_file(self, tmp_path):
        path = str(tmp_path / "validation_report.md")
        render_validation_report(self._make_inputs(), path)
        assert Path(path).exists()

    def test_contains_all_metric_values(self, tmp_path):
        path = str(tmp_path / "validation_report.md")
        inputs = self._make_inputs()
        render_validation_report(inputs, path)
        content = Path(path).read_text()
        assert "0.9500" in content  # lyrics_coverage_pct
        assert "0.8800" in content  # theme_coverage_pct
        assert "0.0800" in content  # low_confidence_theme_rate

    def test_gate_pass_in_report(self, tmp_path):
        path = str(tmp_path / "validation_report.md")
        render_validation_report(self._make_inputs(gate_pass=True), path)
        content = Path(path).read_text()
        assert "PASS" in content

    def test_gate_fail_in_report(self, tmp_path):
        path = str(tmp_path / "validation_report.md")
        render_validation_report(self._make_inputs(gate_pass=False), path)
        content = Path(path).read_text()
        assert "FAIL" in content

    def test_decade_in_report(self, tmp_path):
        path = str(tmp_path / "validation_report.md")
        render_validation_report(self._make_inputs(), path)
        content = Path(path).read_text()
        assert "1990s" in content

    def test_model_versions_in_report(self, tmp_path):
        path = str(tmp_path / "validation_report.md")
        render_validation_report(self._make_inputs(), path)
        content = Path(path).read_text()
        assert "nli-MiniLM2-L6-H768" in content


# ── write_ndjson ──────────────────────────────────────────────────────────────

class TestWriteNdjson:
    def test_one_line_per_record(self, tmp_path, merged_df_full):
        path = str(tmp_path / "output.json")
        write_ndjson(merged_df_full, path)
        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == len(merged_df_full)

    def test_each_line_valid_json(self, tmp_path, merged_df_full):
        path = str(tmp_path / "output.json")
        write_ndjson(merged_df_full, path)
        for line in Path(path).read_text().strip().split("\n"):
            record = json.loads(line)
            assert isinstance(record, dict)

    def test_song_id_present_in_each_record(self, tmp_path, merged_df_full):
        path = str(tmp_path / "output.json")
        write_ndjson(merged_df_full, path)
        for line in Path(path).read_text().strip().split("\n"):
            record = json.loads(line)
            assert "song_id" in record

    def test_null_values_serialised_as_null(self, tmp_path):
        row = _make_merged_row("abc123def456abcd", jungian_primary=None)
        row["jungian_primary"] = None
        df = pd.DataFrame([row])
        path = str(tmp_path / "output.json")
        write_ndjson(df, path)
        record = json.loads(Path(path).read_text().strip())
        assert record["jungian_primary"] is None

    def test_creates_parent_directory(self, tmp_path, merged_df_full):
        path = str(tmp_path / "nested" / "output.json")
        write_ndjson(merged_df_full, path)
        assert Path(path).exists()


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def _base_config(self, minimal_config, tmp_path):
        config = copy.deepcopy(minimal_config)
        config.checkpoints.dir = str(tmp_path / "checkpoints")
        config.outputs.dir = str(tmp_path / "outputs")
        config.outputs.viz_dir = str(tmp_path / "outputs" / "viz")
        config.outputs.report_filename = "validation_report.md"
        config.outputs.analysis_filename = "analysis_{decade}.json"
        config.logging.missing_lyrics_log = str(
            tmp_path / "missing_lyrics.jsonl"
        )
        config.logging.low_confidence_log = str(
            tmp_path / "low_confidence.jsonl"
        )
        Path(config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        Path(config.outputs.dir).mkdir(parents=True, exist_ok=True)
        Path(config.outputs.viz_dir).mkdir(parents=True, exist_ok=True)
        config.checkpoints.force_rerun.s5_report = True
        return config

    def test_returns_dict_with_gate_pass(
        self, minimal_config, tmp_path, merged_df_full
    ):
        config = self._base_config(minimal_config, tmp_path)
        result = run(config, merged_df=merged_df_full, run_id="test_run")
        assert "gate_pass" in result

    def test_all_four_plots_written(
        self, minimal_config, tmp_path, merged_df_full
    ):
        config = self._base_config(minimal_config, tmp_path)
        run(config, merged_df=merged_df_full, run_id="test_run")
        viz_dir = Path(config.outputs.viz_dir)
        for fname in [
            "sentiment_drift.png",
            "mood_heatmap.png",
            "theme_frequency.png",
            "jungian_distribution.png",
        ]:
            assert (viz_dir / fname).exists(), f"Missing viz: {fname}"

    def test_validation_report_written(
        self, minimal_config, tmp_path, merged_df_full
    ):
        config = self._base_config(minimal_config, tmp_path)
        run(config, merged_df=merged_df_full, run_id="test_run")
        report = Path(config.outputs.dir) / config.outputs.report_filename
        assert report.exists()

    def test_ndjson_output_written(
        self, minimal_config, tmp_path, merged_df_full
    ):
        config = self._base_config(minimal_config, tmp_path)
        run(config, merged_df=merged_df_full, run_id="test_run")
        ndjson = Path(config.outputs.dir) / "analysis_1990s.json"
        assert ndjson.exists()

    def test_ndjson_record_count_matches(
        self, minimal_config, tmp_path, merged_df_full
    ):
        config = self._base_config(minimal_config, tmp_path)
        run(config, merged_df=merged_df_full, run_id="test_run")
        ndjson = Path(config.outputs.dir) / "analysis_1990s.json"
        lines = ndjson.read_text().strip().split("\n")
        assert len(lines) == len(merged_df_full)

    def test_gate_pass_true_on_full_data(
        self, minimal_config, tmp_path, merged_df_full
    ):
        config = self._base_config(minimal_config, tmp_path)
        result = run(config, merged_df=merged_df_full, run_id="test_run")
        assert result["gate_pass"] is True

    def test_gate_pass_false_on_low_coverage(
        self, minimal_config, tmp_path, merged_df_low_coverage
    ):
        config = self._base_config(minimal_config, tmp_path)
        result = run(config, merged_df=merged_df_low_coverage, run_id="test_run")
        assert result["gate_pass"] is False

    def test_checkpoint_skips_regeneration_when_report_exists(
        self, minimal_config, tmp_path, merged_df_full
    ):
        config = self._base_config(minimal_config, tmp_path)

        # First run — generates outputs
        run(config, merged_df=merged_df_full, run_id="test_run")

        # Touch run count tracker
        call_count = {"n": 0}
        original_render = render_validation_report

        def counting_render(inputs, path):
            call_count["n"] += 1
            return original_render(inputs, path)

        # Second run with force_rerun=False — must skip
        config.checkpoints.force_rerun.s5_report = False
        from unittest.mock import patch
        with patch("src.stages.s5_report.render_validation_report",
                   side_effect=counting_render):
            run(config, merged_df=merged_df_full, run_id="test_run")

        assert call_count["n"] == 0
