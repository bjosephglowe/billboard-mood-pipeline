import os
import sys
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator


# ── Sub-models ────────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    name: str
    version: str
    year_range: List[int]
    full_year_range: List[int]
    top_n: int

    @field_validator("year_range", "full_year_range")
    @classmethod
    def validate_year_range(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("year_range must be a two-element list [start, end]")
        start, end = v
        if start < 1958 or end > 2024:
            raise ValueError(
                f"year_range [{start}, {end}] is outside supported bounds [1958, 2024]"
            )
        if start > end:
            raise ValueError(f"year_range start {start} must not exceed end {end}")
        return v

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"top_n must be between 1 and 100, got {v}")
        return v


class DatasetConfig(BaseModel):
    github_url: str
    commit_sha: str
    local_path: str


class LyricsConfig(BaseModel):
    genius_sleep_time: float
    musixmatch_truncation_flag: bool
    cache_enabled: bool


class ModelsConfig(BaseModel):
    sentiment: str
    mood: str
    theme: str
    semantic_embedding: str


class InferenceConfig(BaseModel):
    batch_size: int
    sleep_between_batches: float
    device: str
    cache_enabled: bool

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1 or v > 256:
            raise ValueError(f"batch_size must be between 1 and 256, got {v}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ("auto", "mps", "cpu"):
            raise ValueError(f"device must be one of auto | mps | cpu, got {v}")
        return v


class ThemeConfig(BaseModel):
    min_confidence: float
    haiku_fallback_enabled: bool
    semantic_vector_enabled: bool

    @field_validator("min_confidence")
    @classmethod
    def validate_min_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {v}")
        return v


class JungianConfig(BaseModel):
    haiku_model: str
    max_retries: int
    retry_sleep: float


class SemanticConfig(BaseModel):
    tfidf_max_features: int
    tfidf_top_k_keywords: int
    subject_focus_min_pronouns: int
    mtld_min_tokens: int


class CacheConfig(BaseModel):
    lyrics_dir: str
    inference_dir: str


class ForceRerunConfig(BaseModel):
    s1_ingest: bool = False
    s2_lyrics: bool = False
    s3_sentiment: bool = False
    s3_mood: bool = False
    s3_theme: bool = False
    s3_jungian: bool = False
    s3_semantic: bool = False
    s4_merge: bool = False
    s5_report: bool = False


class CheckpointsConfig(BaseModel):
    dir: str
    force_rerun: ForceRerunConfig


class LoggingConfig(BaseModel):
    log_dir: str
    level: str
    missing_lyrics_log: str
    low_confidence_log: str

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"logging.level must be one of {allowed}, got {v}")
        return v.upper()


class OutputsConfig(BaseModel):
    dir: str
    viz_dir: str
    report_filename: str
    analysis_filename: str


class ValidationConfig(BaseModel):
    min_lyrics_coverage: float
    min_sentiment_coverage: float
    min_theme_coverage: float
    min_jungian_coverage: float
    max_low_confidence_theme_rate: float
    min_semantic_coverage: float


# ── Root config model ─────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    project: ProjectConfig
    dataset: DatasetConfig
    lyrics: LyricsConfig
    models: ModelsConfig
    inference: InferenceConfig
    theme: ThemeConfig
    jungian: JungianConfig
    semantic: SemanticConfig
    cache: CacheConfig
    checkpoints: CheckpointsConfig
    logging: LoggingConfig
    outputs: OutputsConfig
    validation: ValidationConfig

    # API keys are loaded from environment, not from YAML.
    # They are attached to the config object after env loading.
    anthropic_api_key: Optional[str] = None
    genius_api_token: Optional[str] = None
    musixmatch_api_key: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ── Required environment keys ─────────────────────────────────────────────────

_REQUIRED_ENV_KEYS = {
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "GENIUS_API_TOKEN": "genius_api_token",
    "MUSIXMATCH_API_KEY": "musixmatch_api_key",
}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(config_path: str, require_env_keys: bool = True) -> PipelineConfig:
    """
    Load and validate config.yaml. Load API keys from .env.

    Args:
        config_path: path to config.yaml
        require_env_keys: if True, abort if any required .env key is missing.
                          Set False only in tests that do not make API calls.

    Returns:
        Validated PipelineConfig instance.

    Raises:
        FileNotFoundError: if config_path does not exist.
        ValueError: if any required env key is missing (when require_env_keys=True).
        pydantic.ValidationError: if config values fail validation.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    config = PipelineConfig(**raw)

    load_dotenv(override=False)

    missing_keys = []
    for env_key, attr_name in _REQUIRED_ENV_KEYS.items():
        value = os.environ.get(env_key)
        if value:
            object.__setattr__(config, attr_name, value)
        elif require_env_keys:
            missing_keys.append(env_key)

    if missing_keys:
        key_list = ", ".join(missing_keys)
        print(
            f"[ERROR] config: Missing required environment variable(s): {key_list}. "
            f"Copy .env.example to .env and populate all keys.",
            file=sys.stderr,
        )
        raise ValueError(
            f"Missing required environment variable(s): {key_list}"
        )

    return config
