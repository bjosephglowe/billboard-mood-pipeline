from pathlib import Path
from typing import Optional

import diskcache

from src.core.config import PipelineConfig
from src.core.logger import get_logger

log = get_logger("cache")


# ── Base cache wrapper ────────────────────────────────────────────────────────

class _BaseCache:
    """
    Thin wrapper around a diskcache.Cache store.

    Provides typed get/set/exists/clear methods with consistent logging.
    Never logs cache values — they may contain full lyric text.
    """

    def __init__(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(path))
        self._directory = directory

    def get(self, key: str) -> Optional[dict]:
        """
        Retrieve a cached value by key.

        Args:
            key: cache key string

        Returns:
            Cached dict, or None if not present.
        """
        value = self._cache.get(key, default=None)
        if value is None:
            log.debug(f"Cache miss: {self._directory} key={key}")
        return value

    def set(self, key: str, value: dict) -> None:
        """
        Store a value in the cache.

        Args:
            key: cache key string
            value: dict to store
        """
        self._cache.set(key, value)

    def exists(self, key: str) -> bool:
        """
        Return True if the key is present in the cache.

        Args:
            key: cache key string
        """
        return key in self._cache

    def clear(self) -> None:
        """
        Remove all entries from the cache.

        This is a destructive operation. Never called by pipeline stages.
        Intended for manual maintenance only.
        """
        self._cache.clear()
        log.warning(f"Cache cleared: {self._directory}")

    def close(self) -> None:
        """Close the underlying diskcache connection."""
        self._cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ── Typed cache classes ───────────────────────────────────────────────────────

class LyricsCache(_BaseCache):
    """
    Cache store for lyrics fetch results.

    Location: data/cache/lyrics/ (from config.cache.lyrics_dir)
    Key pattern: {song_id}
    Value: dict with lyrics, lyrics_source, lyrics_truncated, fetched_at fields.

    Persists permanently across runs. Never auto-cleared by the pipeline.
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config.cache.lyrics_dir)


class InferenceCache(_BaseCache):
    """
    Cache store for model inference results.

    Location: data/cache/inference/ (from config.cache.inference_dir)
    Key pattern: {song_id}_{task}
      where task is one of: sentiment | mood | theme | jungian | semantic

    Persists permanently across runs. Primarily useful for Haiku API calls
    to avoid re-billing on reruns.
    """

    _VALID_TASKS = frozenset({"sentiment", "mood", "theme", "jungian", "semantic"})

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config.cache.inference_dir)

    @staticmethod
    def make_key(song_id: str, task: str) -> str:
        """
        Construct a namespaced cache key for an inference result.

        Args:
            song_id: 16-char hex song identifier
            task: one of sentiment | mood | theme | jungian | semantic

        Returns:
            Key string e.g. "a3f8c1d24e7b9051_sentiment"

        Raises:
            ValueError: if task is not in the allowed set
        """
        if task not in InferenceCache._VALID_TASKS:
            raise ValueError(
                f"InferenceCache: task must be one of {InferenceCache._VALID_TASKS}, "
                f"got '{task}'"
            )
        return f"{song_id}_{task}"

    def get_inference(self, song_id: str, task: str) -> Optional[dict]:
        """Retrieve a cached inference result using the namespaced key."""
        return self.get(self.make_key(song_id, task))

    def set_inference(self, song_id: str, task: str, value: dict) -> None:
        """Store an inference result using the namespaced key."""
        self.set(self.make_key(song_id, task), value)

    def exists_inference(self, song_id: str, task: str) -> bool:
        """Return True if an inference result is cached for this song and task."""
        return self.exists(self.make_key(song_id, task))


# ── Factory functions ─────────────────────────────────────────────────────────

def get_lyrics_cache(config: PipelineConfig) -> LyricsCache:
    """Return a LyricsCache instance backed by config.cache.lyrics_dir."""
    return LyricsCache(config)


def get_inference_cache(config: PipelineConfig) -> InferenceCache:
    """Return an InferenceCache instance backed by config.cache.inference_dir."""
    return InferenceCache(config)
