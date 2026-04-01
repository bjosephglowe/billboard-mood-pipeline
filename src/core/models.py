import time
from typing import Any, Optional, Tuple

from src.core.config import PipelineConfig
from src.core.logger import get_logger

log = get_logger("models")


# ── Device detection ──────────────────────────────────────────────────────────

def get_device(config: PipelineConfig) -> str:
    """
    Determine the compute device to use for transformer inference.

    Respects config.inference.device:
      "auto" — use MPS if available, fall back to CPU
      "mps"  — force MPS; warns if not available and falls back to CPU
      "cpu"  — always use CPU

    Logs the selected device at INFO.

    Args:
        config: PipelineConfig instance

    Returns:
        Device string: "mps" or "cpu"
    """
    import torch

    requested = config.inference.device

    if requested == "cpu":
        log.info("Device: cpu (forced by config)")
        return "cpu"

    mps_available = torch.backends.mps.is_available()

    if requested == "mps":
        if mps_available:
            log.info("Device: mps (forced by config)")
            return "mps"
        else:
            log.warning(
                "Device 'mps' requested in config but MPS is not available. "
                "Falling back to cpu."
            )
            return "cpu"

    # auto
    if mps_available:
        log.info("Device: mps (auto-detected)")
        return "mps"
    else:
        log.info("Device: cpu (MPS not available, auto fallback)")
        return "cpu"


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_pipeline(
    task: str,
    model_name: str,
    device: str,
    extra_kwargs: Optional[dict] = None,
) -> Any:
    """
    Load a HuggingFace transformers pipeline for the given task and model.

    Args:
        task: HuggingFace pipeline task string e.g. "text-classification"
        model_name: HuggingFace model identifier
        device: "mps" or "cpu"
        extra_kwargs: additional kwargs passed to pipeline()

    Returns:
        transformers.Pipeline instance
    """
    from transformers import pipeline

    device_arg = 0 if device == "mps" else -1
    kwargs = extra_kwargs or {}

    log.info(f"Loading model: {model_name} (task={task}, device={device})")
    start = time.time()

    pipe = pipeline(task, model=model_name, device=device_arg, **kwargs)

    elapsed = time.time() - start
    log.info(f"Model loaded in {elapsed:.1f}s: {model_name}")

    return pipe


def load_sentiment_model(config: PipelineConfig, device: str) -> Tuple[Any, str]:
    """
    Load the sentiment classification pipeline.

    Model: config.models.sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest)
    Task: text-classification with top_k=None to get all class probabilities.

    Args:
        config: PipelineConfig instance
        device: device string from get_device()

    Returns:
        Tuple of (pipeline, device)
    """
    pipe = _load_pipeline(
        task="text-classification",
        model_name=config.models.sentiment,
        device=device,
        extra_kwargs={"top_k": None},
    )
    return pipe, device


def load_mood_model(config: PipelineConfig, device: str) -> Tuple[Any, str]:
    """
    Load the mood/emotion classification pipeline.

    Model: config.models.mood (j-hartmann/emotion-english-distilroberta-base)
    Task: text-classification with top_k=None to get all class probabilities.

    Args:
        config: PipelineConfig instance
        device: device string from get_device()

    Returns:
        Tuple of (pipeline, device)
    """
    pipe = _load_pipeline(
        task="text-classification",
        model_name=config.models.mood,
        device=device,
        extra_kwargs={"top_k": None},
    )
    return pipe, device


def load_theme_model(config: PipelineConfig, device: str) -> Tuple[Any, str]:
    """
    Load the zero-shot theme classification pipeline.

    Model: config.models.theme (cross-encoder/nli-MiniLM-L6-v2)
    Task: zero-shot-classification

    Args:
        config: PipelineConfig instance
        device: device string from get_device()

    Returns:
        Tuple of (pipeline, device)
    """
    pipe = _load_pipeline(
        task="zero-shot-classification",
        model_name=config.models.theme,
        device=device,
    )
    return pipe, device


def unload_model(model: Any) -> None:
    """
    Explicitly unload a model and free MPS/CPU memory.

    Deletes the pipeline object and calls torch MPS cache empty if available.
    Must be called after each stage's inference pass completes to prevent
    two transformer models from being resident simultaneously.

    Args:
        model: transformers pipeline instance to unload
    """
    import gc

    import torch

    model_name = getattr(getattr(model, "model", None), "name_or_path", "unknown")
    del model
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    log.debug(f"Model unloaded: {model_name}")
