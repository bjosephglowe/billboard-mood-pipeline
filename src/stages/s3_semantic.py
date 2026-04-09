import re
from typing import Optional

import pandas as pd

from src.core.checkpoint import (
    checkpoint_exists,
    corpus_checkpoint_exists,
    read_checkpoint,
    read_corpus_checkpoint,
    write_checkpoint,
    write_corpus_checkpoint,
)
from src.core.config import PipelineConfig
from src.core.logger import get_logger

log = get_logger("s3_semantic")

# ── Schema 7 columns (locked P5) ─────────────────────────────────────────────
_SCHEMA_7_COLUMNS = [
    "song_id",
    "mtld_score",
    "imagery_density",
    "avg_line_length",
    "tfidf_keywords",
    "subject_focus",
    "semantic_vector",
]

# ── Subject focus pronoun classes (locked P2 Section 6) ──────────────────────
_FIRST_PERSON = frozenset({
    "i", "me", "my", "mine", "myself"
})
_RELATIONAL = frozenset({
    "you", "your", "yours", "yourself",
    "we", "us", "our", "ours", "ourselves"
})
_SOCIETAL = frozenset({
    "they", "them", "their", "theirs", "themselves",
    "people", "world", "everyone", "everybody", "someone"
})

# ── Section header pattern (locked G3 Recommendation 2) ──────────────────────
# Lines matching this pattern are stripped before any analysis.
_SECTION_HEADER_RE = re.compile(r"^\[.*\]$", re.MULTILINE)


# ── Text preprocessing ────────────────────────────────────────────────────────

def preprocess_lyrics(lyrics: str) -> str:
    """
    Strip section headers and normalize lyric text for analysis.

    Rules (locked G3 Recommendation 2):
      1. Strip lines matching ^\\[.*\\]$ (section headers e.g. [Chorus])
      2. Split on newline
      3. Discard blank lines (empty after strip)

    Returns the cleaned text as a single string with lines joined by newline.

    Args:
        lyrics: raw lyric string

    Returns:
        Cleaned lyric string with section headers removed and blank
        lines discarded.
    """
    if not lyrics:
        return ""

    cleaned = _SECTION_HEADER_RE.sub("", lyrics)
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    return "\n".join(lines)


# ── Semantic dimension functions ──────────────────────────────────────────────

def compute_mtld(lyrics: str, min_tokens: int) -> Optional[float]:
    """
    Compute the MTLD (Measure of Textual Lexical Diversity) score.

    Returns None if the token count is strictly less than min_tokens.
    A token count exactly equal to min_tokens is sufficient for computation.

    Args:
        lyrics: preprocessed lyric string
        min_tokens: minimum token count required (strict less-than check)

    Returns:
        MTLD score as float, or None if text is too short.
    """
    from lexicalrichness import LexicalRichness

    tokens = lyrics.split()
    if len(tokens) < min_tokens:
        return None

    try:
        lex = LexicalRichness(lyrics)
        return round(float(lex.mtld(threshold=0.72)), 4)
    except Exception as exc:
        log.debug(f"MTLD computation failed: {exc}")
        return None


def compute_imagery_density(lyrics: str) -> Optional[float]:
    """
    Compute imagery density as the ratio of concrete noun chunks to
    total tokens.

    Args:
        lyrics: preprocessed lyric string

    Returns:
        Float in range 0.0 to 1.0, or None if lyrics is empty.
    """
    if not lyrics or not lyrics.strip():
        return None

    try:
        nlp = _get_spacy_model()
        doc = nlp(lyrics)

        total_tokens = len([t for t in doc if not t.is_space])
        if total_tokens == 0:
            return None

        chunk_token_count = sum(len(chunk) for chunk in doc.noun_chunks)
        return round(chunk_token_count / total_tokens, 6)

    except Exception as exc:
        log.debug(f"Imagery density computation failed: {exc}")
        return None


def compute_avg_line_length(lyrics: str) -> Optional[float]:
    """
    Compute mean tokens per lyric line using spaCy tokenizer.

    Line delimiter rules (locked G3 Recommendation 2):
      - Input must already be preprocessed (section headers stripped,
        blank lines removed)
      - Split on newline
      - Tokenize each remaining line with spaCy
      - avg_line_length = mean of per-line token counts

    Args:
        lyrics: preprocessed lyric string (section headers already stripped)

    Returns:
        Mean token count per line as float, or None if no lines.
    """
    if not lyrics or not lyrics.strip():
        return None

    lines = [line for line in lyrics.split("\n") if line.strip()]
    if not lines:
        return None

    try:
        nlp = _get_spacy_model()
        token_counts = []
        for line in lines:
            doc = nlp(line)
            count = len([t for t in doc if not t.is_space])
            if count > 0:
                token_counts.append(count)

        if not token_counts:
            return None

        return round(sum(token_counts) / len(token_counts), 4)

    except Exception as exc:
        log.debug(f"avg_line_length computation failed: {exc}")
        return None


def compute_subject_focus(
    lyrics: str,
    min_pronouns: int,
) -> str:
    """
    Classify the dominant lyric subject as self, relationship, society,
    mixed, or unknown.

    Method (locked P2 Section 6):
      - Count first-person tokens (I, me, my, mine, myself)
      - Count relational tokens (you, your, we, us, our, ...)
      - Count societal tokens (they, them, people, world, everyone, ...)
      - If total pronoun count < min_pronouns return "unknown"
      - Dominant class > 50% of total pronoun tokens wins
      - No dominant class return "mixed"

    Args:
        lyrics: preprocessed lyric string
        min_pronouns: minimum total pronoun count for a non-unknown result

    Returns:
        One of: "self" | "relationship" | "society" | "mixed" | "unknown"
    """
    if not lyrics or not lyrics.strip():
        return "unknown"

    tokens = [t.lower() for t in lyrics.split()]

    first_count = sum(1 for t in tokens if t in _FIRST_PERSON)
    relational_count = sum(1 for t in tokens if t in _RELATIONAL)
    societal_count = sum(1 for t in tokens if t in _SOCIETAL)

    total = first_count + relational_count + societal_count

    if total < min_pronouns:
        return "unknown"

    threshold = total * 0.50

    if first_count > threshold:
        return "self"
    if relational_count > threshold:
        return "relationship"
    if societal_count > threshold:
        return "society"
    return "mixed"


def score_song_tfidf(
    lyrics: str,
    vectorizer,
    top_k: int,
) -> list[str]:
    """
    Extract the top-k TF-IDF keywords for a single song relative to
    the fitted decade corpus.

    Args:
        lyrics: preprocessed lyric string
        vectorizer: fitted sklearn TfidfVectorizer
        top_k: number of top keywords to return

    Returns:
        List of up to top_k keyword strings. Empty list if lyrics empty.
    """
    import numpy as np

    if not lyrics or not lyrics.strip():
        return []

    try:
        tfidf_matrix = vectorizer.transform([lyrics])
        scores = tfidf_matrix.toarray()[0]

        if scores.sum() == 0:
            return []

        feature_names = vectorizer.get_feature_names_out()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [feature_names[i] for i in top_indices if scores[i] > 0]

    except Exception as exc:
        log.debug(f"TF-IDF scoring failed: {exc}")
        return []


# ── spaCy model singleton ─────────────────────────────────────────────────────

_spacy_nlp = None


def _get_spacy_model():
    """
    Return the spaCy en_core_web_sm model, loading it once per process.
    """
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def _reset_spacy_model():
    """Reset the spaCy singleton. Used in tests to ensure clean state."""
    global _spacy_nlp
    _spacy_nlp = None


# ── TF-IDF corpus build ───────────────────────────────────────────────────────

def build_tfidf_corpus(
    lyrics_list: list[str],
    config: PipelineConfig,
):
    """
    Pass 1: fit a TF-IDF vectorizer on the full decade lyric corpus.

    Filters out None/empty entries before fitting. Raises ValueError if
    no valid lyrics exist.

    Args:
        lyrics_list: list of preprocessed lyric strings (may contain None)
        config: PipelineConfig instance

    Returns:
        Fitted sklearn TfidfVectorizer instance.

    Raises:
        ValueError: if no valid (non-null, non-empty) lyrics are present.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    valid_lyrics = [l for l in lyrics_list if l and l.strip()]

    if not valid_lyrics:
        raise ValueError(
            "Cannot build TF-IDF corpus: no valid lyrics in dataset. "
            "Check that S2 completed successfully and lyrics were fetched."
        )

    log.info(
        f"S3e: building TF-IDF corpus from {len(valid_lyrics)} "
        f"documents (max_features={config.semantic.tfidf_max_features})"
    )

    vectorizer = TfidfVectorizer(
        max_features=config.semantic.tfidf_max_features,
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )
    vectorizer.fit(valid_lyrics)
    log.info("S3e: TF-IDF corpus fit complete.")
    return vectorizer


def _build_null_record(song_id: str) -> dict:
    """Return an all-null Schema 7 record for songs with missing lyrics."""
    return {
        "song_id": song_id,
        "mtld_score": None,
        "imagery_density": None,
        "avg_line_length": None,
        "tfidf_keywords": None,
        "subject_focus": None,
        "semantic_vector": None,
    }


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    lyrics_df: Optional[pd.DataFrame] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 3e: semantic analysis for all songs.

    Two-pass operation (locked G2-R2):
      Pass 1: build and persist TF-IDF corpus from all non-null lyrics.
              Skipped if corpus checkpoint already exists and
              force_rerun.s3_semantic is False.
      Pass 2: compute all six semantic dimensions per song.

    If all songs have missing lyrics, Pass 1 is skipped entirely and all
    records receive null tfidf_keywords — the stage does not abort.

    semantic_vector is always null in validation run (G2-OI3).

    Writes 03_semantic.parquet and 03_tfidf_corpus_{decade}.pkl.

    Args:
        config: PipelineConfig instance
        lyrics_df: optional pre-loaded S2 DataFrame
        run_id: pipeline_run_id for log records

    Returns:
        pd.DataFrame matching Schema 7.
    """
    stage = "s3_semantic"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s3_semantic:
        log.info("S3e: checkpoint found — skipping semantic analysis.")
        return read_checkpoint(stage, config)

    log.info("S3e: starting semantic analysis.")

    if lyrics_df is None:
        lyrics_df = read_checkpoint("s2_lyrics", config)

    decade = (
        str(lyrics_df["decade"].dropna().iloc[0])
        if "decade" in lyrics_df.columns and not lyrics_df["decade"].dropna().empty
        else "unknown"
    )

    # ── Pass 1: TF-IDF corpus build ───────────────────────────────────────────
    vectorizer = None

    if corpus_checkpoint_exists(decade, config) and not config.checkpoints.force_rerun.s3_semantic:
        log.info(f"S3e: corpus checkpoint found for decade={decade} — skipping Pass 1.")
        vectorizer = read_corpus_checkpoint(decade, config)
    else:
        raw_lyrics = lyrics_df["lyrics"].tolist()
        preprocessed = [
            preprocess_lyrics(l) if l else None
            for l in raw_lyrics
        ]
        valid_count = sum(1 for l in preprocessed if l and l.strip())

        if valid_count == 0:
            log.warning(
                "S3e: no valid lyrics found — skipping TF-IDF corpus build. "
                "tfidf_keywords will be null for all records."
            )
            vectorizer = None
        else:
            log.info(f"S3e: Pass 1 — building TF-IDF corpus for decade={decade}.")
            vectorizer = build_tfidf_corpus(preprocessed, config)
            write_corpus_checkpoint(decade, vectorizer, config)

    # ── Pass 2: per-song semantic analysis ────────────────────────────────────
    log.info("S3e: Pass 2 — computing semantic dimensions.")

    records = []
    total = len(lyrics_df)
    null_mtld_count = 0
    unknown_focus_count = 0

    for i, row in enumerate(lyrics_df.to_dict("records"), start=1):
        song_id = row["song_id"]
        lyrics = row.get("lyrics")
        lyrics_status = row.get("lyrics_status", "missing")

        log.debug(
            f"S3e: [{i}/{total}] '{row['title']}' by '{row['artist']}'"
        )

        # ── Null path: missing lyrics ──
        if lyrics_status == "missing" or not lyrics:
            records.append(_build_null_record(song_id))
            continue

        # ── Preprocess ──
        cleaned = preprocess_lyrics(lyrics)

        if not cleaned.strip():
            records.append(_build_null_record(song_id))
            continue

        # ── Compute dimensions ──
        mtld = compute_mtld(cleaned, config.semantic.mtld_min_tokens)
        if mtld is None:
            null_mtld_count += 1

        imagery = compute_imagery_density(cleaned)
        avg_line = compute_avg_line_length(cleaned)

        keywords = None
        if vectorizer is not None:
            kw_list = score_song_tfidf(
                lyrics=cleaned,
                vectorizer=vectorizer,
                top_k=config.semantic.tfidf_top_k_keywords,
            )
            keywords = kw_list if kw_list else None

        focus = compute_subject_focus(
            lyrics=cleaned,
            min_pronouns=config.semantic.subject_focus_min_pronouns,
        )
        if focus == "unknown":
            unknown_focus_count += 1

        records.append({
            "song_id": song_id,
            "mtld_score": mtld,
            "imagery_density": imagery,
            "avg_line_length": avg_line,
            "tfidf_keywords": keywords,
            "subject_focus": focus,
            "semantic_vector": None,  # disabled for validation run (G2-OI3)
        })

    log.info(
        f"S3e: semantic analysis complete. "
        f"total={total} "
        f"null_mtld={null_mtld_count} "
        f"unknown_subject_focus={unknown_focus_count}"
    )

    df = pd.DataFrame(records, columns=_SCHEMA_7_COLUMNS)
    write_checkpoint(stage, df, config)
    log.info("S3e: checkpoint written.")

    return df
