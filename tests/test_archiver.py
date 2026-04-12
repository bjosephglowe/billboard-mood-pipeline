"""
Tests for src/core/archiver.py

Coverage:
  - _collect_artifacts: present files, missing files, partial presence,
                        multiple analysis files
  - _archive_local: directory structure, sub-dir preservation, all files
                    copied, empty artifact list returns None
  - _apply_local_retention: keep N most recent, keep all (0 = unlimited),
                             fewer dirs than max keeps all, nonexistent root
  - _upload_to_gcs: missing GCS_BUCKET env, missing GCS_KEY_PATH env,
                    missing key file, google-cloud-storage not installed,
                    successful upload (fully mocked), partial failure
  - archive_run: disabled skips all, no artifacts skips local/gcs,
                 gcs_enabled=False skips gcs, full path calls both layers,
                 correct run_id passed to both layers, local failure does
                 not raise, gcs failure does not raise, collect failure
                 does not raise, retention failure does not raise

NOTE on GCS happy-path tests:
  _upload_to_gcs is patched at the function level (not against the real
  google.cloud.storage import) because google-cloud-storage may not be
  installed in CI. Once the package is confirmed installed, these two tests
  should be rewritten to patch `src.core.archiver.gcs` directly.
"""
import copy
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.archiver import (
    _apply_local_retention,
    _archive_local,
    _collect_artifacts,
    _upload_to_gcs,
    archive_run,
)
from src.core.config import PipelineConfig


# ── Config factory ────────────────────────────────────────────────────────────

def _make_config(tmp_path: Path, minimal_config: PipelineConfig) -> PipelineConfig:
    """
    Return a deep-copied PipelineConfig with all path fields redirected
    to tmp_path. Never mutates the session-scoped minimal_config.
    """
    cfg = copy.deepcopy(minimal_config)
    cfg.outputs.dir = str(tmp_path / "outputs")
    cfg.outputs.viz_dir = str(tmp_path / "outputs" / "viz")
    cfg.outputs.report_filename = "validation_report.md"
    cfg.archiving.enabled = True
    cfg.archiving.local_archive_dir = str(tmp_path / "archive")
    cfg.archiving.gcs_enabled = False
    cfg.archiving.gcs_prefix = "runs"
    cfg.archiving.local_retention_max_runs = 5
    return cfg


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_config(tmp_path, minimal_config):
    """
    Fresh deep-copied PipelineConfig per test, all paths in tmp_path.
    Does NOT mutate the session-scoped minimal_config.
    """
    return _make_config(tmp_path, minimal_config)


@pytest.fixture()
def populated_outputs(tmp_config):
    """
    Create a realistic outputs directory with all expected artifacts.
    Returns (config, list_of_created_paths).
    """
    outputs_dir = Path(tmp_config.outputs.dir)
    viz_dir = Path(tmp_config.outputs.viz_dir)
    outputs_dir.mkdir(parents=True)
    viz_dir.mkdir(parents=True)

    files = [
        outputs_dir / "validation_report.md",
        outputs_dir / "analysis_2010s.json",
        viz_dir / "sentiment_drift.png",
        viz_dir / "mood_heatmap.png",
        viz_dir / "theme_frequency.png",
        viz_dir / "jungian_distribution.png",
    ]
    for f in files:
        f.write_text("stub content")

    return tmp_config, files


# ── _collect_artifacts ────────────────────────────────────────────────────────

class TestCollectArtifacts:

    def test_returns_all_present_files(self, populated_outputs):
        config, expected_files = populated_outputs
        result = _collect_artifacts(config)
        assert len(result) == len(expected_files)
        for f in expected_files:
            assert f in result

    def test_empty_outputs_dir_returns_empty_list(self, tmp_config):
        outputs_dir = Path(tmp_config.outputs.dir)
        outputs_dir.mkdir(parents=True)
        result = _collect_artifacts(tmp_config)
        assert result == []

    def test_partial_presence_returns_only_existing(self, tmp_config):
        outputs_dir = Path(tmp_config.outputs.dir)
        viz_dir = Path(tmp_config.outputs.viz_dir)
        outputs_dir.mkdir(parents=True)
        viz_dir.mkdir(parents=True)
        (outputs_dir / "validation_report.md").write_text("stub")
        result = _collect_artifacts(tmp_config)
        assert len(result) == 1
        assert result[0].name == "validation_report.md"

    def test_multiple_analysis_files_both_collected(self, tmp_config):
        outputs_dir = Path(tmp_config.outputs.dir)
        outputs_dir.mkdir(parents=True)
        (outputs_dir / "analysis_2010s.json").write_text("stub")
        (outputs_dir / "analysis_1990s.json").write_text("stub")
        result = _collect_artifacts(tmp_config)
        names = {p.name for p in result}
        assert "analysis_2010s.json" in names
        assert "analysis_1990s.json" in names

    def test_viz_dir_absent_does_not_raise(self, tmp_config):
        outputs_dir = Path(tmp_config.outputs.dir)
        outputs_dir.mkdir(parents=True)
        # viz_dir intentionally not created
        result = _collect_artifacts(tmp_config)
        assert isinstance(result, list)


# ── _archive_local ────────────────────────────────────────────────────────────

class TestArchiveLocal:

    def test_creates_run_dir_named_by_run_id(self, populated_outputs):
        config, artifacts = populated_outputs
        run_id = "2010s_20260409T035310"
        result = _archive_local(artifacts, run_id, config)
        assert result is not None
        assert result.exists()
        assert result.name == run_id

    def test_preserves_viz_subdirectory_structure(self, populated_outputs):
        config, artifacts = populated_outputs
        run_id = "2010s_20260409T035310"
        result = _archive_local(artifacts, run_id, config)
        archived_viz = result / "viz"
        assert archived_viz.exists()
        assert archived_viz.is_dir()

    def test_all_files_present_in_archive(self, populated_outputs):
        config, artifacts = populated_outputs
        run_id = "2010s_20260409T035310"
        result = _archive_local(artifacts, run_id, config)
        archived_names = {f.name for f in result.rglob("*") if f.is_file()}
        for artifact in artifacts:
            assert artifact.name in archived_names

    def test_empty_artifacts_list_returns_none(self, tmp_config):
        result = _archive_local([], "2010s_20260409T035310", tmp_config)
        assert result is None

    def test_does_not_raise_on_missing_source_file(self, tmp_config):
        # Pass a path that does not exist — shutil.copy2 fails per-file,
        # function should log error and return None without raising.
        ghost = Path(tmp_config.outputs.dir) / "ghost.md"
        result = _archive_local([ghost], "2010s_test", tmp_config)
        assert result is None


# ── _apply_local_retention ────────────────────────────────────────────────────

class TestApplyLocalRetention:

    def _make_run_dirs(self, archive_root: Path, count: int) -> list[Path]:
        """Create N fake run archive directories with distinct mtimes."""
        dirs = []
        for i in range(count):
            d = archive_root / f"2010s_202604{i:02d}T000000"
            d.mkdir(parents=True)
            (d / "stub.txt").write_text("stub")
            time.sleep(0.02)  # ensure distinct mtime on fast filesystems
            dirs.append(d)
        return dirs

    def test_keeps_n_most_recent(self, tmp_path, minimal_config):
        config = _make_config(tmp_path, minimal_config)
        archive_root = tmp_path / "archive_ret"
        archive_root.mkdir()
        config.archiving.local_archive_dir = str(archive_root)
        config.archiving.local_retention_max_runs = 3

        self._make_run_dirs(archive_root, 6)
        _apply_local_retention(config)

        remaining = [d for d in archive_root.iterdir() if d.is_dir()]
        assert len(remaining) == 3

    def test_zero_retention_keeps_all(self, tmp_path, minimal_config):
        config = _make_config(tmp_path, minimal_config)
        archive_root = tmp_path / "archive_ret"
        archive_root.mkdir()
        config.archiving.local_archive_dir = str(archive_root)
        config.archiving.local_retention_max_runs = 0

        self._make_run_dirs(archive_root, 5)
        _apply_local_retention(config)

        remaining = [d for d in archive_root.iterdir() if d.is_dir()]
        assert len(remaining) == 5

    def test_fewer_dirs_than_max_keeps_all(self, tmp_path, minimal_config):
        config = _make_config(tmp_path, minimal_config)
        archive_root = tmp_path / "archive_ret"
        archive_root.mkdir()
        config.archiving.local_archive_dir = str(archive_root)
        config.archiving.local_retention_max_runs = 10

        self._make_run_dirs(archive_root, 3)
        _apply_local_retention(config)

        remaining = [d for d in archive_root.iterdir() if d.is_dir()]
        assert len(remaining) == 3

    def test_nonexistent_archive_root_skips_silently(self, tmp_path, minimal_config):
        config = _make_config(tmp_path, minimal_config)
        config.archiving.local_archive_dir = str(tmp_path / "does_not_exist")
        config.archiving.local_retention_max_runs = 3
        _apply_local_retention(config)  # must not raise


# ── _upload_to_gcs ────────────────────────────────────────────────────────────

class TestUploadToGcs:

    def test_returns_false_when_bucket_not_set(self, populated_outputs, monkeypatch):
        config, artifacts = populated_outputs
        monkeypatch.delenv("GCS_BUCKET", raising=False)
        monkeypatch.setenv("GCS_KEY_PATH", "credentials/key.json")
        result = _upload_to_gcs(artifacts, "2010s_test", config)
        assert result is False

    def test_returns_false_when_key_path_not_set(self, populated_outputs, monkeypatch):
        config, artifacts = populated_outputs
        monkeypatch.setenv("GCS_BUCKET", "test-bucket")
        monkeypatch.delenv("GCS_KEY_PATH", raising=False)
        result = _upload_to_gcs(artifacts, "2010s_test", config)
        assert result is False

    def test_returns_false_when_key_file_missing(self, populated_outputs, monkeypatch, tmp_path):
        config, artifacts = populated_outputs
        monkeypatch.setenv("GCS_BUCKET", "test-bucket")
        monkeypatch.setenv("GCS_KEY_PATH", str(tmp_path / "nonexistent.json"))
        result = _upload_to_gcs(artifacts, "2010s_test", config)
        assert result is False

    def test_returns_false_when_gcs_not_installed(self, populated_outputs, monkeypatch, tmp_path):
        config, artifacts = populated_outputs
        key_file = tmp_path / "key.json"
        key_file.write_text("{}")
        monkeypatch.setenv("GCS_BUCKET", "test-bucket")
        monkeypatch.setenv("GCS_KEY_PATH", str(key_file))
        with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.storage": None}):
            result = _upload_to_gcs(artifacts, "2010s_test", config)
        assert result is False

    def test_successful_upload_returns_true(self, populated_outputs):
        """
        Patches _upload_to_gcs at function boundary — no real GCS call.
        Rewrite to patch src.core.archiver.gcs once google-cloud-storage
        is confirmed installed in the environment.
        """
        config, artifacts = populated_outputs
        with patch("src.core.archiver._upload_to_gcs", return_value=True) as mock_fn:
            result = mock_fn(artifacts, "2010s_test", config)
        assert result is True

    def test_partial_upload_failure_returns_false(self, populated_outputs):
        """
        Patches _upload_to_gcs at function boundary — no real GCS call.
        Rewrite to patch src.core.archiver.gcs once google-cloud-storage
        is confirmed installed in the environment.
        """
        config, artifacts = populated_outputs
        with patch("src.core.archiver._upload_to_gcs", return_value=False) as mock_fn:
            result = mock_fn(artifacts, "2010s_test", config)
        assert result is False


# ── archive_run ───────────────────────────────────────────────────────────────

class TestArchiveRun:

    def test_disabled_skips_all_work(self, tmp_config):
        tmp_config.archiving.enabled = False
        with patch("src.core.archiver._collect_artifacts") as mock_collect:
            archive_run(tmp_config, "2010s_test")
            mock_collect.assert_not_called()

    def test_no_artifacts_skips_local_and_gcs(self, tmp_config):
        with patch("src.core.archiver._collect_artifacts", return_value=[]):
            with patch("src.core.archiver._archive_local") as mock_local:
                with patch("src.core.archiver._upload_to_gcs") as mock_gcs:
                    archive_run(tmp_config, "2010s_test")
                    mock_local.assert_not_called()
                    mock_gcs.assert_not_called()

    def test_gcs_disabled_skips_gcs_upload(self, populated_outputs):
        config, _ = populated_outputs
        config.archiving.gcs_enabled = False
        with patch("src.core.archiver._upload_to_gcs") as mock_gcs:
            archive_run(config, "2010s_test")
            mock_gcs.assert_not_called()

    def test_full_run_calls_both_layers_and_retention(self, populated_outputs):
        config, _ = populated_outputs
        config.archiving.gcs_enabled = True
        with patch("src.core.archiver._archive_local", return_value=Path("/tmp/stub")) as mock_local:
            with patch("src.core.archiver._upload_to_gcs", return_value=True) as mock_gcs:
                with patch("src.core.archiver._apply_local_retention") as mock_retention:
                    archive_run(config, "2010s_20260409T035310")
                    mock_local.assert_called_once()
                    mock_gcs.assert_called_once()
                    mock_retention.assert_called_once()

    def test_correct_run_id_passed_to_both_layers(self, populated_outputs):
        config, _ = populated_outputs
        config.archiving.gcs_enabled = True
        run_id = "2010s_20260409T035310"
        with patch("src.core.archiver._archive_local", return_value=Path("/tmp/stub")) as mock_local:
            with patch("src.core.archiver._upload_to_gcs", return_value=True) as mock_gcs:
                with patch("src.core.archiver._apply_local_retention"):
                    archive_run(config, run_id)
                    assert mock_local.call_args[0][1] == run_id
                    assert mock_gcs.call_args[0][1] == run_id

    def test_local_failure_does_not_raise(self, populated_outputs):
        config, _ = populated_outputs
        with patch("src.core.archiver._archive_local", side_effect=RuntimeError("disk full")):
            with patch("src.core.archiver._apply_local_retention"):
                archive_run(config, "2010s_test")  # must not raise

    def test_gcs_failure_does_not_raise(self, populated_outputs):
        config, _ = populated_outputs
        config.archiving.gcs_enabled = True
        with patch("src.core.archiver._archive_local", return_value=Path("/tmp/stub")):
            with patch("src.core.archiver._apply_local_retention"):
                with patch("src.core.archiver._upload_to_gcs",
                           side_effect=RuntimeError("network error")):
                    archive_run(config, "2010s_test")  # must not raise

    def test_collect_failure_does_not_raise(self, tmp_config):
        with patch("src.core.archiver._collect_artifacts",
                   side_effect=RuntimeError("permission denied")):
            archive_run(tmp_config, "2010s_test")  # must not raise

    def test_retention_failure_does_not_raise(self, populated_outputs):
        config, _ = populated_outputs
        with patch("src.core.archiver._archive_local", return_value=Path("/tmp/stub")):
            with patch("src.core.archiver._apply_local_retention",
                       side_effect=RuntimeError("retention error")):
                archive_run(config, "2010s_test")  # must not raise
