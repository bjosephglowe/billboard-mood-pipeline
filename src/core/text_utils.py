"""
Shared text utilities used by multiple analysis stages.

Placed in src/core/ to avoid inter-stage module imports. Any stage
that needs text preprocessing should import from here, never from
another stage module.
"""


def chunk_lyrics(lyrics: str, max_tokens: int = 512) -> list[str]:
    """
    Split lyrics into chunks that fit within a model's token limit.

    Uses a word-based approximation (1 token ≈ 0.75 words) to avoid
    loading a tokenizer solely for chunking. Chunks on line boundaries
    where possible to preserve lyric structure.

    Used by: s3_sentiment.py, s3_mood.py, s3_theme.py

    Args:
        lyrics: full lyrics string
        max_tokens: maximum tokens per chunk (default 512)

    Returns:
        List of lyric chunk strings. Empty list if lyrics is empty.
    """
    if not lyrics or not lyrics.strip():
        return []

    # Approximate word limit: max_tokens * 0.75 words/token
    max_words = int(max_tokens * 0.75)

    lines = [line for line in lyrics.split("\n") if line.strip()]
    chunks: list[str] = []
    current_lines: list[str] = []
    current_word_count = 0

    for line in lines:
        line_words = len(line.split())
        if current_word_count + line_words > max_words and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_word_count = line_words
        else:
            current_lines.append(line)
            current_word_count += line_words

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks if chunks else [lyrics]
