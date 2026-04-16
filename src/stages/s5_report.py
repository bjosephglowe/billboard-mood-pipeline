import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.checkpoint import checkpoint_exists, read_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger

log = get_logger("s5_report")

# ── Validation threshold keys (locked P2 Section 8) ──────────────────────────
# Maps config field names to the metric keys they gate.
_THRESHOLD_MAP = {
    "min_lyrics_coverage":          "lyrics_coverage_pct",
    "min_sentiment_coverage":       "sentiment_coverage_pct",
    "min_theme_coverage":           "theme_coverage_pct",
    "min_jungian_coverage":         "jungian_coverage_pct",
    "max_low_confidence_theme_rate":"low_confidence_theme_rate",
    "min_semantic_coverage":        "semantic_coverage_pct",
}


# ── Schema 11: validation report inputs ──────────────────────────────────────

def compute_report_inputs(
    df: pd.DataFrame,
    config: PipelineConfig,
    run_id: str,
) -> dict:
    """
    Compute all Schema 11 validation report input fields from the
    merged DataFrame.

    All coverage percentages are computed as fractions (0.0 to 1.0),
    not percentages (0 to 100), to match the config threshold values.

    Args:
        df: Schema 8 merged DataFrame
        config: PipelineConfig instance
        run_id: pipeline_run_id string

    Returns:
        Dict matching Schema 11 (locked P5).
    """
    import platform
    import sys

    total = len(df)

    # ── Lyrics coverage ──
    lyrics_found = int((df["lyrics_status"] == "found").sum())
    lyrics_truncated = int((df["lyrics_status"] == "truncated").sum())
    lyrics_missing = int((df["lyrics_status"] == "missing").sum())
    lyrics_coverage = (lyrics_found + lyrics_truncated) / total if total else 0.0

    # ── Sentiment coverage ──
    sentiment_coverage = (
        int(df["sentiment_score"].notna().sum()) / total if total else 0.0
    )

    # ── Theme coverage ──
    theme_non_uncertain = int(
        (df["theme_primary"].notna() & (df["theme_primary"] != "uncertain")).sum()
    )
    theme_coverage = theme_non_uncertain / total if total else 0.0

    # ── Jungian coverage ──
    jungian_coverage = (
        int(df["jungian_primary"].notna().sum()) / total if total else 0.0
    )

    # ── Semantic coverage: all 5 required fields populated ──
    semantic_required = [
        "mtld_score", "imagery_density", "avg_line_length",
        "tfidf_keywords", "subject_focus",
    ]
    semantic_complete = df[semantic_required].notna().all(axis=1)
    semantic_coverage = int(semantic_complete.sum()) / total if total else 0.0

    # ── Low-confidence theme rate ──
    # Counts songs where theme classification failed entirely
    # (theme_primary == "uncertain"). Songs classified via Haiku fallback
    # carry valid theme labels and are NOT counted as low-confidence.
    # Redefined per Issue 1 Option B — see build state snapshot.
    low_conf_theme = int(
        (df["theme_primary"] == "uncertain").sum()
    )
    low_conf_theme_rate = low_conf_theme / total if total else 0.0

    # ── record_complete ──
    record_complete_count = int(df["record_complete"].sum())

    # ── Skipped songs ──
    skipped_count = int(df["skip_reason"].notna().sum())

    # ── Distributions ──
    theme_labels = [
        "love_and_romance", "heartbreak_and_loss", "identity_and_self",
        "rebellion_and_defiance", "spirituality_and_faith",
        "materialism_and_ambition", "nostalgia_and_memory",
        "social_commentary", "hedonism_and_pleasure", "longing_and_desire",
        "conflict_and_struggle", "celebration_and_joy", "uncertain",
    ]
    mood_labels = [
        "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"
    ]
    archetype_labels = [
        "hero", "shadow", "anima_animus", "self", "trickster",
        "great_mother", "wise_old_man", "persona",
    ]

    theme_dist = {
        label: int((df["theme_primary"] == label).sum())
        for label in theme_labels
    }
    mood_dist = {
        label: int((df["mood_primary"] == label).sum())
        for label in mood_labels
    }
    jungian_dist = {
        label: int((df["jungian_primary"] == label).sum())
        for label in archetype_labels
    }

    # ── Sentiment stats ──
    scored = df["sentiment_score"].dropna()
    sentiment_mean = round(float(scored.mean()), 6) if len(scored) > 0 else None
    sentiment_std = round(float(scored.std()), 6) if len(scored) > 1 else None

    # ── Model versions ──
    model_versions = {
        "sentiment": config.models.sentiment,
        "mood":      config.models.mood,
        "theme":     config.models.theme,
        "jungian":   config.jungian.haiku_model,
        "semantic":  "spacy:en_core_web_sm + sklearn:TfidfVectorizer",
    }

    # ── Decade ──
    decade = str(df["decade"].dropna().iloc[0]) if not df["decade"].dropna().empty else "unknown"
    year_range = config.project.year_range

    return {
        "run_id":                     run_id,
        "decade":                     decade,
        "year_range":                 year_range,
        "total_songs":                total,
        "lyrics_found_count":         lyrics_found,
        "lyrics_truncated_count":     lyrics_truncated,
        "lyrics_missing_count":       lyrics_missing,
        "lyrics_coverage_pct":        round(lyrics_coverage, 4),
        "sentiment_coverage_pct":     round(sentiment_coverage, 4),
        "theme_coverage_pct":         round(theme_coverage, 4),
        "jungian_coverage_pct":       round(jungian_coverage, 4),
        "semantic_coverage_pct":      round(semantic_coverage, 4),
        "low_confidence_theme_rate":  round(low_conf_theme_rate, 4),
        "record_complete_count":      record_complete_count,
        "record_complete_pct":        round(record_complete_count / total, 4) if total else 0.0,
        "skipped_count":              skipped_count,
        "theme_distribution":         theme_dist,
        "mood_distribution":          mood_dist,
        "jungian_distribution":       jungian_dist,
        "sentiment_mean":             sentiment_mean,
        "sentiment_std":              sentiment_std,
        "model_versions":             model_versions,
        "pipeline_run_timestamp":     run_id,
        "gate_pass":                  False,  # set after evaluate_gate()
    }


def evaluate_gate(inputs: dict, config: PipelineConfig) -> bool:
    """
    Evaluate all P2 validation thresholds against computed metrics.

    Locked threshold checks (P2 Section 8):
      - lyrics_coverage_pct        >= config.validation.min_lyrics_coverage
      - sentiment_coverage_pct     >= config.validation.min_sentiment_coverage
      - theme_coverage_pct         >= config.validation.min_theme_coverage
      - jungian_coverage_pct       >= config.validation.min_jungian_coverage
      - semantic_coverage_pct      >= config.validation.min_semantic_coverage
      - low_confidence_theme_rate  <= config.validation.max_low_confidence_theme_rate

    Logs each threshold result at INFO. Logs failures at WARNING.

    Args:
        inputs: dict from compute_report_inputs()
        config: PipelineConfig instance

    Returns:
        True if all thresholds pass, False otherwise.
    """
    v = config.validation
    checks = [
        ("lyrics_coverage_pct",
         inputs["lyrics_coverage_pct"],
         v.min_lyrics_coverage, ">="),
        ("sentiment_coverage_pct",
         inputs["sentiment_coverage_pct"],
         v.min_sentiment_coverage, ">="),
        ("theme_coverage_pct",
         inputs["theme_coverage_pct"],
         v.min_theme_coverage, ">="),
        ("jungian_coverage_pct",
         inputs["jungian_coverage_pct"],
         v.min_jungian_coverage, ">="),
        ("semantic_coverage_pct",
         inputs["semantic_coverage_pct"],
         v.min_semantic_coverage, ">="),
        ("low_confidence_theme_rate",
         inputs["low_confidence_theme_rate"],
         v.max_low_confidence_theme_rate, "<="),
    ]

    all_pass = True
    for metric, value, threshold, op in checks:
        if op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold

        status = "PASS" if passed else "FAIL"
        msg = (
            f"S5: gate check [{status}] "
            f"{metric}={value:.4f} {op} {threshold:.4f}"
        )
        if passed:
            log.info(msg)
        else:
            log.warning(msg)
            all_pass = False

    return all_pass


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_sentiment_drift(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot mean sentiment score per year as a line chart.

    Args:
        df: Schema 8 merged DataFrame
        output_path: file path to write PNG
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    scored = df[df["sentiment_score"].notna()].copy()
    if scored.empty:
        log.warning("S5: no sentiment scores available — skipping sentiment drift plot.")
        return

    yearly = (
        scored.groupby("year")["sentiment_score"]
        .mean()
        .reset_index()
        .sort_values("year")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        yearly["year"], yearly["sentiment_score"],
        marker="o", linewidth=1.5, markersize=4, color="#1D9E75",
    )
    ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Mean sentiment score", fontsize=11)
    ax.set_title("Sentiment drift by year", fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(-1.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info(f"S5: sentiment drift chart written: {output_path}")


def plot_mood_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot a heatmap of mood label counts per year.

    Args:
        df: Schema 8 merged DataFrame
        output_path: file path to write PNG
    """
    import matplotlib.pyplot as plt

    mood_labels = [
        "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"
    ]
    scored = df[df["mood_primary"].notna()].copy()
    if scored.empty:
        log.warning("S5: no mood data available — skipping mood heatmap.")
        return

    pivot = (
        scored.groupby(["year", "mood_primary"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=mood_labels, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.4 + 1)))
    im = ax.imshow(
        pivot.values.T,
        aspect="auto",
        cmap="YlGnBu",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mood_labels)))
    ax.set_yticklabels(mood_labels, fontsize=9)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title("Mood distribution by year", fontsize=13)
    fig.colorbar(im, ax=ax, label="Song count")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info(f"S5: mood heatmap written: {output_path}")


def plot_theme_frequency(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot theme label frequency as a horizontal bar chart.

    Args:
        df: Schema 8 merged DataFrame
        output_path: file path to write PNG
    """
    import matplotlib.pyplot as plt

    scored = df[
        df["theme_primary"].notna() & (df["theme_primary"] != "uncertain")
    ].copy()
    if scored.empty:
        log.warning("S5: no theme data available — skipping theme frequency plot.")
        return

    counts = scored["theme_primary"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.45 + 1)))
    ax.barh(counts.index, counts.values, color="#534AB7", alpha=0.85)
    ax.set_xlabel("Song count", fontsize=11)
    ax.set_title("Theme frequency", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info(f"S5: theme frequency chart written: {output_path}")


def plot_jungian_distribution(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot Jungian archetype distribution as a bar chart.

    Args:
        df: Schema 8 merged DataFrame
        output_path: file path to write PNG
    """
    import matplotlib.pyplot as plt

    scored = df[df["jungian_primary"].notna()].copy()
    if scored.empty:
        log.warning("S5: no Jungian data available — skipping archetype distribution plot.")
        return

    counts = scored["jungian_primary"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color="#D85A30", alpha=0.85)
    ax.set_xlabel("Archetype", fontsize=11)
    ax.set_ylabel("Song count", fontsize=11)
    ax.set_title("Jungian archetype distribution", fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info(f"S5: Jungian distribution chart written: {output_path}")


# ── Validation report ─────────────────────────────────────────────────────────

def render_validation_report(inputs: dict, output_path: str) -> None:
    """
    Render the validation report markdown file from Schema 11 inputs.

    Args:
        inputs: dict from compute_report_inputs() with gate_pass set
        output_path: file path to write markdown
    """
    gate_str = "PASS" if inputs["gate_pass"] else "FAIL"
    yr = inputs["year_range"]

    lines = [
        f"# Billboard Cultural Mood — Validation Report",
        f"",
        f"**Run ID:** {inputs['run_id']}  ",
        f"**Decade:** {inputs['decade']}  ",
        f"**Year range:** {yr[0]}–{yr[1]}  ",
        f"**Total songs:** {inputs['total_songs']}  ",
        f"**Gate result:** {gate_str}  ",
        f"",
        f"---",
        f"",
        f"## Coverage metrics",
        f"",
        f"| Metric | Value | Threshold | Status |",
        f"|--------|-------|-----------|--------|",
    ]

    threshold_rows = [
        ("Lyrics coverage",
         inputs["lyrics_coverage_pct"],
         "≥ 0.85", inputs["lyrics_coverage_pct"] >= 0.85),
        ("Sentiment coverage",
         inputs["sentiment_coverage_pct"],
         "≥ 0.85", inputs["sentiment_coverage_pct"] >= 0.85),
        ("Theme coverage",
         inputs["theme_coverage_pct"],
         "≥ 0.75", inputs["theme_coverage_pct"] >= 0.75),
        ("Jungian coverage",
         inputs["jungian_coverage_pct"],
         "≥ 0.60", inputs["jungian_coverage_pct"] >= 0.60),
        ("Semantic coverage",
         inputs["semantic_coverage_pct"],
         "≥ 0.85", inputs["semantic_coverage_pct"] >= 0.85),
        ("Low-confidence theme rate",
         inputs["low_confidence_theme_rate"],
         "≤ 0.25", inputs["low_confidence_theme_rate"] <= 0.25),
    ]

    for name, value, threshold, passed in threshold_rows:
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"| {name} | {value:.4f} | {threshold} | {status} |")

    lines += [
        f"",
        f"## Record completeness",
        f"",
        f"- Complete records: {inputs['record_complete_count']} "
        f"({inputs['record_complete_pct']:.1%})",
        f"- Skipped songs: {inputs['skipped_count']}",
        f"- Songs with missing lyrics: {inputs['lyrics_missing_count']}",
        f"- Songs with truncated lyrics: {inputs['lyrics_truncated_count']}",
        f"",
        f"## Sentiment",
        f"",
        f"- Mean score: {inputs['sentiment_mean']}"
        if inputs['sentiment_mean'] is not None
        else "- Mean score: N/A",
        f"- Std deviation: {inputs['sentiment_std']}"
        if inputs['sentiment_std'] is not None
        else "- Std deviation: N/A",
        f"",
        f"## Theme distribution",
        f"",
    ]

    for label, count in sorted(
        inputs["theme_distribution"].items(), key=lambda x: -x[1]
    ):
        if count > 0:
            lines.append(f"- {label}: {count}")

    lines += [
        f"",
        f"## Mood distribution",
        f"",
    ]
    for label, count in sorted(
        inputs["mood_distribution"].items(), key=lambda x: -x[1]
    ):
        if count > 0:
            lines.append(f"- {label}: {count}")

    lines += [
        f"",
        f"## Jungian archetype distribution",
        f"",
    ]
    for label, count in sorted(
        inputs["jungian_distribution"].items(), key=lambda x: -x[1]
    ):
        if count > 0:
            lines.append(f"- {label}: {count}")

    lines += [
        f"",
        f"## Model versions",
        f"",
    ]
    for task, model in inputs["model_versions"].items():
        lines.append(f"- {task}: `{model}`")

    content = "\n".join(lines) + "\n"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info(f"S5: validation report written: {output_path}")


# ── NDJSON output ─────────────────────────────────────────────────────────────

def _serialise_row(row: dict) -> dict:
    import numpy as np
    out = {}
    for k, v in row.items():
        if isinstance(v, (list, dict)):
            out[k] = v
        elif isinstance(v, np.ndarray):
            # numpy arrays from parquet (e.g. jungian_evidence) — convert to list
            out[k] = v.tolist() if v.size > 0 else None
        elif hasattr(v, "item"):
            # numpy scalar — convert to Python native
            out[k] = v.item()
        else:
            try:
                out[k] = None if pd.isna(v) else v
            except (TypeError, ValueError):
                out[k] = v
    return out


def write_ndjson(df: pd.DataFrame, output_path: str) -> None:
    """
    Serialise the merged DataFrame to newline-delimited JSON.

    Each record is one JSON object per line. This is the primary
    structured output consumed by downstream tools.

    Args:
        df: Schema 8 merged DataFrame
        output_path: file path to write NDJSON
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            serialised = _serialise_row(row)
            f.write(json.dumps(serialised, ensure_ascii=False) + "\n")
    log.info(f"S5: NDJSON output written: {output_path} ({len(df)} records)")


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    merged_df: Optional[pd.DataFrame] = None,
    run_id: str = "unknown",
) -> dict:
    """
    Execute Stage 5: produce all visualizations, NDJSON output, and
    validation report.

    Reads S4 checkpoint if merged_df not provided. Computes all Schema 11
    report inputs. Evaluates gate pass/fail against P2 thresholds.
    Writes four PNG visualizations, one NDJSON output, and one markdown
    validation report.

    Args:
        config: PipelineConfig instance
        merged_df: optional pre-loaded Schema 8 DataFrame
        run_id: pipeline_run_id for report header

    Returns:
        Schema 11 inputs dict with gate_pass populated.
    """
    stage = "s5_report"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s5_report:
        log.info("S5: outputs found — skipping report generation.")
        # Read and return inputs from existing report for gate_pass value
        report_path = Path(config.outputs.dir) / config.outputs.report_filename
        log.info(f"S5: existing report at {report_path}")
        return {}

    log.info("S5: starting report generation.")

    if merged_df is None:
        merged_df = read_checkpoint("s4_merge", config)

    decade = str(merged_df["decade"].dropna().iloc[0]) \
        if not merged_df["decade"].dropna().empty else "unknown"

    # ── Compute report inputs ──
    inputs = compute_report_inputs(merged_df, config, run_id)

    # ── Evaluate gate ──
    inputs["gate_pass"] = evaluate_gate(inputs, config)

    # ── Write visualizations ──
    viz_dir = config.outputs.viz_dir
    plot_sentiment_drift(
        merged_df,
        str(Path(viz_dir) / "sentiment_drift.png"),
    )
    plot_mood_heatmap(
        merged_df,
        str(Path(viz_dir) / "mood_heatmap.png"),
    )
    plot_theme_frequency(
        merged_df,
        str(Path(viz_dir) / "theme_frequency.png"),
    )
    plot_jungian_distribution(
        merged_df,
        str(Path(viz_dir) / "jungian_distribution.png"),
    )

    # ── Write NDJSON output ──
    analysis_filename = config.outputs.analysis_filename.format(decade=decade)
    write_ndjson(
        merged_df,
        str(Path(config.outputs.dir) / analysis_filename),
    )

    # ── Write validation report ──
    render_validation_report(
        inputs,
        str(Path(config.outputs.dir) / config.outputs.report_filename),
    )

    gate_str = "PASS" if inputs["gate_pass"] else "FAIL"
    log.info(
        f"S5: report complete. gate={gate_str} "
        f"record_complete={inputs['record_complete_count']}/{inputs['total_songs']}"
    )

    return inputs
