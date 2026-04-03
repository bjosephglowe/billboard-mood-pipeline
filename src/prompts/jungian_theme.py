"""
Locked Haiku prompt template and output schema for combined
Jungian analysis and theme fallback classification.

This module is the single source of truth for:
  - prompt construction (build_prompt)
  - response parsing and validation (parse_response)
  - permitted label sets (get_permitted_archetypes, get_permitted_themes)

Locked in G2-OI2. All field names and permitted values must remain
consistent with 03_jungian.parquet and 03_theme.parquet contracts (P5).

No API call logic lives here. API call mechanics are owned by s3_jungian.py.
"""

import json
from typing import Optional

# ── Permitted label sets (locked P2 Section 5, P5 Schema 6) ──────────────────

_PERMITTED_ARCHETYPES = frozenset({
    "hero",
    "shadow",
    "anima_animus",
    "self",
    "trickster",
    "great_mother",
    "wise_old_man",
    "persona",
})

# Locked P2 Section 4, P5 Schema 5 — excludes "uncertain" which is a
# pipeline-assigned state, not a model output label.
_PERMITTED_THEMES = frozenset({
    "love_and_romance",
    "heartbreak_and_loss",
    "identity_and_self",
    "rebellion_and_defiance",
    "spirituality_and_faith",
    "materialism_and_ambition",
    "nostalgia_and_memory",
    "social_commentary",
    "hedonism_and_pleasure",
    "longing_and_desire",
    "conflict_and_struggle",
    "celebration_and_joy",
})


# ── Public label accessors ────────────────────────────────────────────────────

def get_permitted_archetypes() -> list[str]:
    """Return sorted list of permitted Jungian archetype labels."""
    return sorted(_PERMITTED_ARCHETYPES)


def get_permitted_themes() -> list[str]:
    """Return sorted list of permitted theme labels."""
    return sorted(_PERMITTED_THEMES)


# ── Prompt construction ───────────────────────────────────────────────────────

# Guardrail instructions — must appear verbatim in every built prompt.
# Locked in G2-OI2.
_GUARDRAILS = """\
Rules you must follow:
1. For jungian.primary: only assign a label if you can cite specific lyric phrases as evidence. If you cannot find clear evidence, return null.
2. For jungian.evidence: provide 1 to 3 direct quoted phrases from the lyrics. This field must be non-empty if primary is non-null.
3. For jungian.flag: set to "insufficient_evidence" if primary is null. Set to "speculative" if evidence is indirect or a single weak phrase.
4. For theme_fallback: only populate if requested is true. Select only from the permitted theme labels. Return null if no label fits with reasonable confidence.
5. Return only valid JSON. No preamble, no explanation, no markdown fences."""


def build_prompt(
    lyrics: str,
    song_title: str,
    artist: str,
    request_theme_fallback: bool,
) -> str:
    """
    Build the combined Jungian + optional theme fallback prompt for Claude Haiku.

    The prompt instructs the model to return a single JSON object matching
    the locked G2-OI2 output schema. All five guardrail rules are included
    verbatim.

    Args:
        lyrics: full lyrics text for the song
        song_title: song title (for context only, not used in classification)
        artist: artist name (for context only)
        request_theme_fallback: if True, the theme_fallback block is activated
                                 and the model must attempt theme classification.
                                 if False, theme_fallback.requested is false and
                                 label fields must be null.

    Returns:
        Prompt string ready to send to the Haiku API.
    """
    archetypes_list = ", ".join(sorted(_PERMITTED_ARCHETYPES))
    themes_list = ", ".join(sorted(_PERMITTED_THEMES))
    theme_requested_str = "true" if request_theme_fallback else "false"

    prompt = f"""Analyse the following song lyrics and return a JSON object with exactly this structure:

{{
  "jungian": {{
    "primary": <one of: {archetypes_list}> or null,
    "secondary": <one of: {archetypes_list}> or null,
    "confidence": <one of: "low", "medium", "high"> or null,
    "evidence": [<1 to 3 direct lyric phrases>] or null,
    "flag": <one of: "insufficient_evidence", "speculative"> or null
  }},
  "theme_fallback": {{
    "requested": {theme_requested_str},
    "primary": <one of: {themes_list}> or null,
    "primary_confidence": <float 0.0 to 1.0> or 0.0,
    "secondary": <one of: {themes_list}> or null,
    "secondary_confidence": <float 0.0 to 1.0> or 0.0
  }}
}}

{_GUARDRAILS}

Song: "{song_title}" by {artist}

Lyrics:
{lyrics}"""

    return prompt


# ── Response parsing and validation ──────────────────────────────────────────

class PromptParseError(Exception):
    """Raised when the model response cannot be parsed or fails validation."""
    pass


def parse_response(raw_json: str) -> dict:
    """
    Parse and validate a raw Haiku API response against the locked schema.

    Validates:
      - Response is valid JSON
      - jungian.primary is from permitted archetypes or null
      - jungian.secondary is from permitted archetypes or null
      - jungian.evidence is non-empty list if jungian.primary is non-null
      - jungian.confidence is one of low/medium/high or null
      - jungian.flag is one of the permitted values or null
      - theme_fallback.primary is from permitted themes or null
      - theme_fallback.secondary is from permitted themes or null

    Args:
        raw_json: raw string response from the Haiku API

    Returns:
        Validated dict matching the G2-OI2 output schema.

    Raises:
        PromptParseError: if JSON is invalid or any field fails validation.
    """
    # Strip markdown code fences if present despite guardrail instruction
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        cleaned = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise PromptParseError(
            f"Response is not valid JSON: {exc}\nRaw response: {raw_json[:200]}"
        ) from exc

    # ── Validate jungian block ──
    jungian = data.get("jungian", {})
    if not isinstance(jungian, dict):
        raise PromptParseError("'jungian' field is missing or not an object.")

    j_primary = jungian.get("primary")
    if j_primary is not None and j_primary not in _PERMITTED_ARCHETYPES:
        raise PromptParseError(
            f"jungian.primary '{j_primary}' is not in permitted archetypes: "
            f"{sorted(_PERMITTED_ARCHETYPES)}"
        )

    j_secondary = jungian.get("secondary")
    if j_secondary is not None and j_secondary not in _PERMITTED_ARCHETYPES:
        raise PromptParseError(
            f"jungian.secondary '{j_secondary}' is not in permitted archetypes: "
            f"{sorted(_PERMITTED_ARCHETYPES)}"
        )

    j_evidence = jungian.get("evidence")
    if j_primary is not None:
        if not j_evidence or not isinstance(j_evidence, list) or len(j_evidence) == 0:
            raise PromptParseError(
                f"jungian.evidence must be a non-empty list when "
                f"jungian.primary is non-null. Got: {j_evidence}"
            )

    j_confidence = jungian.get("confidence")
    _permitted_confidence = {None, "low", "medium", "high"}
    if j_confidence not in _permitted_confidence:
        raise PromptParseError(
            f"jungian.confidence '{j_confidence}' must be one of "
            f"{_permitted_confidence}."
        )

    j_flag = jungian.get("flag")
    _permitted_jungian_flags = {None, "insufficient_evidence", "speculative"}
    if j_flag not in _permitted_jungian_flags:
        raise PromptParseError(
            f"jungian.flag '{j_flag}' must be one of {_permitted_jungian_flags}."
        )

    # ── Validate theme_fallback block ──
    theme_fb = data.get("theme_fallback", {})
    if not isinstance(theme_fb, dict):
        raise PromptParseError("'theme_fallback' field is missing or not an object.")

    t_primary = theme_fb.get("primary")
    if t_primary is not None and t_primary not in _PERMITTED_THEMES:
        raise PromptParseError(
            f"theme_fallback.primary '{t_primary}' is not in permitted themes: "
            f"{sorted(_PERMITTED_THEMES)}"
        )

    t_secondary = theme_fb.get("secondary")
    if t_secondary is not None and t_secondary not in _PERMITTED_THEMES:
        raise PromptParseError(
            f"theme_fallback.secondary '{t_secondary}' is not in permitted themes: "
            f"{sorted(_PERMITTED_THEMES)}"
        )

    # ── Normalise and return ──
    # Return a clean dict with all expected keys present and typed correctly,
    # using safe .get() defaults for any fields the model may have omitted.
    return {
        "jungian": {
            "primary": j_primary,
            "secondary": j_secondary,
            "confidence": j_confidence,
            "evidence": j_evidence if j_evidence else None,
            "flag": j_flag,
        },
        "theme_fallback": {
            "requested": bool(theme_fb.get("requested", False)),
            "primary": t_primary,
            "primary_confidence": float(theme_fb.get("primary_confidence") or 0.0),
            "secondary": t_secondary,
            "secondary_confidence": float(
                theme_fb.get("secondary_confidence") or 0.0
            ),
        },
    }
