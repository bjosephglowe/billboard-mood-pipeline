"""
Smoke tests — P4 exit criterion.

Five tests only. Verify that:
  1. config loads from config.yaml without error
  2. all required runtime directories exist
  3. logger writes to the log directory without error
  4. all three required .env keys are present in the environment
  5. get_device() returns a valid device string without error

These tests make no API calls, load no model weights, and perform no
inference. They are the minimum bar for confirming a working environment
before any stage implementation begins.

Run with:
    pytest tests/test_smoke.py -v
"""
import os
from pathlib import Path

import pytest

from src.core.config import load_config
from src.core.logger import get_logger, init_run_logging

# ── Constants ─────────────────────────────────────────────────────────────────

_PROD_CONFIG_PATH = "config/config.yaml"

# Directories that must exist before the pipeline can run.
# Defined in P4 Section 5 and P3 repository structure.
_REQUIRED_DIRS = [
    "config",
    "data/raw",
    "data/cache/lyrics",
    "data/cache/inference",
    "checkpoints",
    "outputs/viz",
    "logs",
    "src/stages",
    "src/core",
    "src/prompts",
    "src/future",
    "tests",
]

# Environment keys required for all API-dependent stages.
_REQUIRED_ENV_KEYS = [
    "ANTHROPIC_API_KEY",
    "GENIUS_API_TOKEN",
    "MUSIXMATCH_API_KEY",
]


# ── Test 1 — Config loads ─────────────────────────────────────────────────────

def test_config_loads():
    """
    config/config.yaml must load and validate without error.

    Uses require_env_keys=False so this test passes on machines where
    .env is not yet populated. Key presence is verified separately in
    test_env_keys_present.
    """
    config = load_config(_PROD_CONFIG_PATH, require_env_keys=False)

    assert config is not None, "load_config returned None"
    assert config.project.name == "billboard-mood-pipeline"
    assert len(config.project.year_range) == 2
    assert config.project.year_range[0] >= 1958
    assert config.project.year_range[1] <= 2024
    assert config.inference.batch_size >= 1
    assert 0.0 <= config.theme.min_confidence <= 1.0


# ── Test 2 — Directories exist ────────────────────────────────────────────────

def test_directories_exist():
    """
    All required runtime directories must exist on disk.

    This test verifies that the P4 directory scaffold was executed
    before implementation began.
    """
    missing = [d for d in _REQUIRED_DIRS if not Path(d).is_dir()]

    assert not missing, (
        f"The following required directories are missing: {missing}\n"
        f"Run the P4 scaffold commands to create them."
    )


# ── Test 3 — Logger writes ────────────────────────────────────────────────────

def test_logger_writes(tmp_path):
    """
    Logger must initialize and write to a log file without error.

    Uses tmp_path (pytest built-in) so test output does not pollute
    the production logs/ directory.
    """
    init_run_logging(
        log_dir=str(tmp_path),
        level="DEBUG",
        run_timestamp="smoke_test",
    )

    log = get_logger("test_smoke")
    log.info("Smoke test log write — INFO")
    log.debug("Smoke test log write — DEBUG")
    log.warning("Smoke test log write — WARNING")

    log_file = tmp_path / "run_smoke_test.log"
    assert log_file.exists(), f"Log file was not created at {log_file}"

    content = log_file.read_text(encoding="utf-8")
    assert "Smoke test log write — INFO" in content
    assert "Smoke test log write — WARNING" in content


# ── Test 4 — Environment keys present ────────────────────────────────────────

def test_env_keys_present():
    """
    All three required API keys must be present as environment variables.

    Does NOT validate that the keys are correct or that APIs are reachable.
    That validation happens during stage S2 and S3d implementation.

    Loads .env via python-dotenv before checking, consistent with how
    load_config() behaves in production.
    """
    from dotenv import load_dotenv
    load_dotenv(override=False)

    missing = [k for k in _REQUIRED_ENV_KEYS if not os.environ.get(k)]

    assert not missing, (
        f"The following required environment variables are not set: {missing}\n"
        f"Copy .env.example to .env and populate all three keys."
    )


# ── Test 5 — MPS / device detected ───────────────────────────────────────────

def test_mps_detected():
    """
    get_device() must return either 'mps' or 'cpu' without raising.

    On a MacBook Air M4, MPS should be available and 'mps' should be
    returned. On any other machine, 'cpu' is the valid fallback.
    Both outcomes are acceptable — this test confirms the detection
    logic runs without error.
    """
    from src.core.config import load_config
    from src.core.models import get_device

    config = load_config(_PROD_CONFIG_PATH, require_env_keys=False)
    device = get_device(config)

    assert device in ("mps", "cpu"), (
        f"get_device() returned unexpected value: '{device}'. "
        f"Expected 'mps' or 'cpu'."
    )
