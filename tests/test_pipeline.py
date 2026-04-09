"""
Tests for src/pipeline.py

Covers:
  - generate_run_id format matches {decade}_{YYYYMMDD}T{HHMMSS}
  - generate_run_id decade derived correctly from year_range
  - write_run_manifest appends two records per run
  - write_run_manifest startup record has null completed_at and gate_pass
  - write_run_manifest completion record has populated completed_at
  - write_run_manifest all Schema 12 fields present
  - apply_force_reruns calls invalidate_from for set flags
  - apply_force_reruns does not call invalidate_from for unset flags
  - run_pipeline calls all stage run() functions
  - run_pipeline stage exception does not abort subsequent stages
  - run_pipeline returns gate_pass from S5
  - run_pipeline returns None when S5 fails
  - run_id passed to stage run() calls
"""
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline import (
    _config_hash,
    apply_force_reruns,
    generate_run_id,
    run_pipeline,
    write_run_manifest,
)

# ── Schema 12 required fields (locked pre-P5) ────────────────────────────────
_SCHEMA_12_FIELDS = [
    "pipeline_run_id",
    "year_range",
    "decade",
    "config_hash",
    "config_path",
    "started_at",
    "completed_at",
    "gate_pass",
    "python_version",
    "platform",
]

_RUN_ID_PATTERN = re.compile(r"^\d{4}s_\d{8}T\d{6}$")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _mock_df():
    return pd.DataFrame([{"song_id": "abc123def456abcd"}])


def _patch_all_stages(mocks: dict):
    """
    Return a single context manager that patches all nine stage run()
    functions simultaneously using nested patches.

    Accepts a dict mapping stage dotted-path strings to mock objects.
    """
    from contextlib import ExitStack

    stack = ExitStack()
    for target, mock in mocks.items():
        stack.enter_context(patch(target, mock))
    return stack


def _make_stage_mocks(gate_pass: bool = True) -> dict:
    mock_df = _mock_df()
    return {
        "src.stages.s1_ingest.run":    MagicMock(return_value=mock_df),
        "src.stages.s2_lyrics.run":    MagicMock(return_value=mock_df),
        "src.stages.s3_sentiment.run": MagicMock(return_value=mock_df),
        "src.stages.s3_mood.run":      MagicMock(return_value=mock_df),
        "src.stages.s3_theme.run":     MagicMock(return_value=mock_df),
        "src.stages.s3_jungian.run":   MagicMock(return_value=mock_df),
        "src.stages.s3_semantic.run":  MagicMock(return_value=mock_df),
        "src.stages.s4_merge.run":     MagicMock(return_value=mock_df),
        "src.stages.s5_report.run":    MagicMock(return_value={"gate_pass": gate_pass}),
    }


# ── generate_run_id ───────────────────────────────────────────────────────────

class TestGenerateRunId:
    def test_format_matches_pattern(self, minimal_config):
        run_id = generate_run_id(minimal_config)
        assert _RUN_ID_PATTERN.match(run_id), (
            f"run_id '{run_id}' does not match pattern {{decade}}_{{YYYYMMDD}}T{{HHMMSS}}"
        )

    def test_decade_1990s(self, minimal_config):
        minimal_config.project.year_range = [1993, 1993]
        assert generate_run_id(minimal_config).startswith("1990s_")

    def test_decade_1950s(self, minimal_config):
        minimal_config.project.year_range = [1958, 1959]
        assert generate_run_id(minimal_config).startswith("1950s_")

    def test_decade_2000s(self, minimal_config):
        minimal_config.project.year_range = [2000, 2009]
        assert generate_run_id(minimal_config).startswith("2000s_")

    def test_timestamp_portion_length(self, minimal_config):
        run_id = generate_run_id(minimal_config)
        ts_part = run_id.split("_", 1)[1]
        assert len(ts_part) == 15

    def test_two_calls_differ(self, minimal_config):
        import time
        id1 = generate_run_id(minimal_config)
        time.sleep(1.1)
        id2 = generate_run_id(minimal_config)
        assert id1 != id2


# ── write_run_manifest ────────────────────────────────────────────────────────

class TestWriteRunManifest:
    def _base_config(self, minimal_config, tmp_path):
        minimal_config.logging.log_dir = str(tmp_path / "logs")
        Path(minimal_config.logging.log_dir).mkdir(parents=True, exist_ok=True)
        return minimal_config

    def test_manifest_file_created(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        write_run_manifest("1990s_test", config, "2024-05-01T14:32:00Z")
        assert (Path(config.logging.log_dir) / "run_manifest.jsonl").exists()

    def test_startup_record_null_completed_at(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        write_run_manifest("1990s_test", config, "2024-05-01T14:32:00Z")
        record = json.loads(
            (Path(config.logging.log_dir) / "run_manifest.jsonl")
            .read_text().strip()
        )
        assert record["completed_at"] is None
        assert record["gate_pass"] is None

    def test_completion_record_populated(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        write_run_manifest(
            "1990s_test", config, "2024-05-01T14:32:00Z",
            completed_at="2024-05-01T16:14:33Z",
            gate_pass=True,
        )
        record = json.loads(
            (Path(config.logging.log_dir) / "run_manifest.jsonl")
            .read_text().strip()
        )
        assert record["completed_at"] == "2024-05-01T16:14:33Z"
        assert record["gate_pass"] is True

    def test_two_writes_append_two_records(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        write_run_manifest("1990s_test", config, "2024-05-01T14:32:00Z")
        write_run_manifest(
            "1990s_test", config, "2024-05-01T14:32:00Z",
            completed_at="2024-05-01T16:14:33Z", gate_pass=True,
        )
        lines = (
            (Path(config.logging.log_dir) / "run_manifest.jsonl")
            .read_text().strip().split("\n")
        )
        assert len(lines) == 2

    def test_all_schema_12_fields_present(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        write_run_manifest("1990s_test", config, "2024-05-01T14:32:00Z")
        record = json.loads(
            (Path(config.logging.log_dir) / "run_manifest.jsonl")
            .read_text().strip()
        )
        for field in _SCHEMA_12_FIELDS:
            assert field in record, f"Missing Schema 12 field: {field}"

    def test_run_id_preserved(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        write_run_manifest("1990s_20240501T143200", config, "2024-05-01T14:32:00Z")
        record = json.loads(
            (Path(config.logging.log_dir) / "run_manifest.jsonl")
            .read_text().strip()
        )
        assert record["pipeline_run_id"] == "1990s_20240501T143200"

    def test_year_range_in_record(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        config.project.year_range = [1990, 1999]
        write_run_manifest("1990s_test", config, "2024-05-01T14:32:00Z")
        record = json.loads(
            (Path(config.logging.log_dir) / "run_manifest.jsonl")
            .read_text().strip()
        )
        assert record["year_range"] == [1990, 1999]

    def test_multiple_runs_all_appended(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        for i in range(3):
            write_run_manifest(f"1990s_run{i}", config, "2024-05-01T14:32:00Z")
        lines = (
            (Path(config.logging.log_dir) / "run_manifest.jsonl")
            .read_text().strip().split("\n")
        )
        assert len(lines) == 3


# ── apply_force_reruns ────────────────────────────────────────────────────────

class TestApplyForceReruns:
    def _base_config(self, minimal_config, tmp_path):
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        return minimal_config

    def test_invalidate_called_for_set_flag(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        config.checkpoints.force_rerun.s3_sentiment = True
        with patch("src.pipeline.invalidate_from") as mock_inv:
            apply_force_reruns(config)
        called_stages = [c.args[0] for c in mock_inv.call_args_list]
        assert "s3_sentiment" in called_stages

    def test_invalidate_not_called_when_all_false(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        for attr in [
            "s1_ingest", "s2_lyrics", "s3_sentiment", "s3_mood",
            "s3_theme", "s3_jungian", "s3_semantic", "s4_merge", "s5_report",
        ]:
            setattr(config.checkpoints.force_rerun, attr, False)
        with patch("src.pipeline.invalidate_from") as mock_inv:
            apply_force_reruns(config)
        mock_inv.assert_not_called()

    def test_decade_passed_to_invalidate(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        config.project.year_range = [1990, 1999]
        config.checkpoints.force_rerun.s3_semantic = True
        with patch("src.pipeline.invalidate_from") as mock_inv:
            apply_force_reruns(config)
        kwargs = mock_inv.call_args_list[0][1]
        assert kwargs.get("decade") == "1990s"

    def test_multiple_flags_all_processed(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        config.checkpoints.force_rerun.s3_sentiment = True
        config.checkpoints.force_rerun.s5_report = True
        with patch("src.pipeline.invalidate_from") as mock_inv:
            apply_force_reruns(config)
        called_stages = [c.args[0] for c in mock_inv.call_args_list]
        assert "s3_sentiment" in called_stages
        assert "s5_report" in called_stages


# ── run_pipeline ──────────────────────────────────────────────────────────────

class TestRunPipeline:
    def _base_config(self, minimal_config, tmp_path):
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        minimal_config.outputs.dir = str(tmp_path / "outputs")
        minimal_config.outputs.viz_dir = str(tmp_path / "outputs" / "viz")
        minimal_config.logging.log_dir = str(tmp_path / "logs")
        minimal_config.logging.missing_lyrics_log = str(
            tmp_path / "logs" / "missing_lyrics.jsonl"
        )
        minimal_config.logging.low_confidence_log = str(
            tmp_path / "logs" / "low_confidence.jsonl"
        )
        for d in [
            tmp_path / "checkpoints",
            tmp_path / "outputs",
            tmp_path / "logs",
        ]:
            d.mkdir(parents=True, exist_ok=True)
        return minimal_config

    def test_all_stages_called(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks()
        with _patch_all_stages(mocks):
            run_pipeline(config, "1990s_test")
        for target, mock in mocks.items():
            assert mock.called, f"{target} was not called"

    def test_returns_gate_pass_true(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks(gate_pass=True)
        with _patch_all_stages(mocks):
            result = run_pipeline(config, "1990s_test")
        assert result is True

    def test_returns_gate_pass_false(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks(gate_pass=False)
        with _patch_all_stages(mocks):
            result = run_pipeline(config, "1990s_test")
        assert result is False

    def test_returns_none_when_s5_fails(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks()
        mocks["src.stages.s5_report.run"].side_effect = Exception("S5 crashed")
        with _patch_all_stages(mocks):
            result = run_pipeline(config, "1990s_test")
        assert result is None

    def test_stage_exception_does_not_abort_subsequent_stages(
        self, minimal_config, tmp_path
    ):
        """S3a failure must not prevent S3b from running."""
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks()
        mocks["src.stages.s3_sentiment.run"].side_effect = Exception("S3a crashed")
        with _patch_all_stages(mocks):
            run_pipeline(config, "1990s_test")
        assert mocks["src.stages.s3_mood.run"].called

    def test_s1_exception_still_runs_remaining_stages(
        self, minimal_config, tmp_path
    ):
        """S1 failure should still allow S2–S5 to attempt execution."""
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks()
        mocks["src.stages.s1_ingest.run"].side_effect = Exception("S1 crashed")
        with _patch_all_stages(mocks):
            run_pipeline(config, "1990s_test")
        assert mocks["src.stages.s2_lyrics.run"].called
        assert mocks["src.stages.s5_report.run"].called

    def test_run_id_passed_to_s2(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks()
        with _patch_all_stages(mocks):
            run_pipeline(config, "1990s_20240501T143200")
        call_kwargs = mocks["src.stages.s2_lyrics.run"].call_args[1]
        assert call_kwargs.get("run_id") == "1990s_20240501T143200"

    def test_run_id_passed_to_s5(self, minimal_config, tmp_path):
        config = self._base_config(minimal_config, tmp_path)
        mocks = _make_stage_mocks()
        with _patch_all_stages(mocks):
            run_pipeline(config, "1990s_20240501T143200")
        call_kwargs = mocks["src.stages.s5_report.run"].call_args[1]
        assert call_kwargs.get("run_id") == "1990s_20240501T143200"
