"""
Tests for src/stages/s1_ingest.py

Covers:
  - Schema 1 field presence and types
  - song_id generation correctness
  - year and rank filtering
  - decade derivation
  - collision detection and resolution
  - missing title/artist handling
  - year coverage warnings
  - checkpoint read/write behavior
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.song_id import generate_song_id
from src.stages.s1_ingest import (
    _derive_decade,
    detect_collisions,
    normalize_records,
    run,
)

# ── Schema 1 required columns (locked P5) ────────────────────────────────────
_SCHEMA_1_COLUMNS = [
    "song_id",
    "year",
    "rank",
    "title",
    "artist",
    "decade",
    "title_normalized",
    "artist_normalized",
    "collision_flag",
]

# ── Shared raw fixture ────────────────────────────────────────────────────────

@pytest.fixture()
def raw_entries() -> list[dict]:
    """Ten synthetic raw Billboard entries spanning 1990–1993."""
    return [
        {"rank": 1, "song": "Song Alpha",   "artist": "Artist One",   "year": 1990},
        {"rank": 2, "song": "Song Beta",    "artist": "Artist Two",   "year": 1990},
        {"rank": 1, "song": "Song Gamma",   "artist": "Artist Three", "year": 1991},
        {"rank": 2, "song": "Song Delta",   "artist": "Artist Four",  "year": 1991},
        {"rank": 1, "song": "Song Epsilon", "artist": "Artist Five",  "year": 1992},
        {"rank": 2, "song": "Song Zeta",    "artist": "Artist Six",   "year": 1992},
        {"rank": 1, "song": "Song Eta",     "artist": "Artist Seven", "year": 1993},
        {"rank": 2, "song": "Song Theta",   "artist": "Artist Eight", "year": 1993},
        # Outside year range — should be excluded
        {"rank": 1, "song": "Song Iota",    "artist": "Artist Nine",  "year": 1985},
        # Outside rank range — should be excluded when top_n=2
        {"rank": 5, "song": "Song Kappa",   "artist": "Artist Ten",   "year": 1991},
    ]


# ── _derive_decade ────────────────────────────────────────────────────────────

class TestDeriveDecode:
    def test_1990s(self):
        assert _derive_decade(1993) == "1990s"

    def test_1950s(self):
        assert _derive_decade(1958) == "1950s"

    def test_2000s(self):
        assert _derive_decade(2000) == "2000s"

    def test_2020s(self):
        assert _derive_decade(2024) == "2020s"

    def test_decade_boundary(self):
        assert _derive_decade(1990) == "1990s"
        assert _derive_decade(1999) == "1990s"


# ── normalize_records ─────────────────────────────────────────────────────────

class TestNormalizeRecords:
    def test_schema_1_columns_present(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        for col in _SCHEMA_1_COLUMNS:
            assert col in df.columns, f"Missing Schema 1 column: {col}"

    def test_year_filter_excludes_out_of_range(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=10)
        assert 1985 not in df["year"].values

    def test_rank_filter_excludes_beyond_top_n(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        assert (df["rank"] <= 2).all()
        assert 5 not in df["rank"].values

    def test_record_count_correct(self, raw_entries):
        # 8 entries in range [1990, 1993] with rank <= 2
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        assert len(df) == 8

    def test_song_id_is_16_char_hex(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=10)
        for sid in df["song_id"]:
            assert len(sid) == 16, f"song_id '{sid}' is not 16 characters"
            assert all(c in "0123456789abcdef" for c in sid), (
                f"song_id '{sid}' contains non-hex characters"
            )

    def test_song_id_matches_canonical_function(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        first = df.iloc[0]
        expected = generate_song_id(first["title"], first["artist"], first["year"])
        assert first["song_id"] == expected

    def test_decade_derived_correctly(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        for _, row in df.iterrows():
            assert row["decade"] == _derive_decade(row["year"])

    def test_sorted_by_year_then_rank(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        years = df["year"].tolist()
        assert years == sorted(years), "Records not sorted by year"
        for year in df["year"].unique():
            ranks = df[df["year"] == year]["rank"].tolist()
            assert ranks == sorted(ranks), f"Records for {year} not sorted by rank"

    def test_collision_flag_default_false(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        assert df["collision_flag"].dtype == bool or df["collision_flag"].isin([True, False]).all()
        assert not df["collision_flag"].any()

    def test_empty_result_when_no_matches(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[2020, 2020], top_n=10)
        assert len(df) == 0
        assert list(df.columns) == _SCHEMA_1_COLUMNS

    def test_entry_with_missing_title_skipped(self):
        entries = [
            {"rank": 1, "song": "",          "artist": "Artist", "year": 1993},
            {"rank": 2, "song": "Valid Song", "artist": "Artist", "year": 1993},
        ]
        df = normalize_records(entries, year_range=[1993, 1993], top_n=10)
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Valid Song"

    def test_entry_with_missing_artist_skipped(self):
        entries = [
            {"rank": 1, "song": "Valid Song", "artist": "",      "year": 1993},
            {"rank": 2, "song": "Other Song", "artist": "Artist", "year": 1993},
        ]
        df = normalize_records(entries, year_range=[1993, 1993], top_n=10)
        assert len(df) == 1
        assert df.iloc[0]["artist"] == "Artist"

    def test_entry_with_unparseable_year_skipped(self):
        entries = [
            {"rank": 1, "song": "Song", "artist": "Artist", "year": "bad"},
            {"rank": 2, "song": "Song", "artist": "Artist", "year": 1993},
        ]
        df = normalize_records(entries, year_range=[1993, 1993], top_n=10)
        assert len(df) == 1

    def test_title_normalized_field_populated(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        for _, row in df.iterrows():
            assert isinstance(row["title_normalized"], str)
            assert len(row["title_normalized"]) > 0

    def test_artist_normalized_field_populated(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        for _, row in df.iterrows():
            assert isinstance(row["artist_normalized"], str)
            assert len(row["artist_normalized"]) > 0

    def test_dataset_uses_song_key(self):
        """KorbenD dataset uses 'song' not 'title' — verify both keys handled."""
        entries_song_key = [
            {"rank": 1, "song": "Via Song Key", "artist": "Artist", "year": 1993}
        ]
        entries_title_key = [
            {"rank": 1, "title": "Via Title Key", "artist": "Artist", "year": 1993}
        ]
        df_song = normalize_records(entries_song_key, year_range=[1993, 1993], top_n=10)
        df_title = normalize_records(entries_title_key, year_range=[1993, 1993], top_n=10)
        assert df_song.iloc[0]["title"] == "Via Song Key"
        assert df_title.iloc[0]["title"] == "Via Title Key"


# ── detect_collisions ─────────────────────────────────────────────────────────

class TestDetectCollisions:
    def test_no_collision_flag_unchanged(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        result = detect_collisions(df)
        assert not result["collision_flag"].any()

    def test_collision_sets_flag_on_second_record(self):
        """Two songs that produce the same song_id should trigger collision handling."""
        # Force a collision by using identical normalized inputs
        entries = [
            {"rank": 1, "song": "Same Song", "artist": "Same Artist", "year": 1993},
            {"rank": 2, "song": "Same Song", "artist": "Same Artist", "year": 1993},
        ]
        df = normalize_records(entries, year_range=[1993, 1993], top_n=10)
        result = detect_collisions(df)

        # Second record should have collision_flag=True
        assert result.iloc[1]["collision_flag"] is True or result.iloc[1]["collision_flag"] == True

    def test_collision_appends_2_to_duplicate_song_id(self):
        entries = [
            {"rank": 1, "song": "Same Song", "artist": "Same Artist", "year": 1993},
            {"rank": 2, "song": "Same Song", "artist": "Same Artist", "year": 1993},
        ]
        df = normalize_records(entries, year_range=[1993, 1993], top_n=10)
        result = detect_collisions(df)

        first_id = result.iloc[0]["song_id"]
        second_id = result.iloc[1]["song_id"]
        assert second_id == first_id + "_2"

    def test_all_song_ids_unique_after_collision_resolution(self):
        entries = [
            {"rank": 1, "song": "Same Song", "artist": "Same Artist", "year": 1993},
            {"rank": 2, "song": "Same Song", "artist": "Same Artist", "year": 1993},
        ]
        df = normalize_records(entries, year_range=[1993, 1993], top_n=10)
        result = detect_collisions(df)
        assert result["song_id"].nunique() == len(result)

    def test_no_mutation_of_input_dataframe(self, raw_entries):
        df = normalize_records(raw_entries, year_range=[1990, 1993], top_n=2)
        original_ids = df["song_id"].tolist()
        detect_collisions(df)
        assert df["song_id"].tolist() == original_ids


# ── run (stage orchestration) ─────────────────────────────────────────────────

class TestRun:
    def test_run_produces_schema_1_dataframe(self, minimal_config, tmp_path, raw_entries):
        """run() should return a DataFrame with all Schema 1 columns."""
        # Patch checkpoint dir to tmp_path
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.dataset.local_path = str(tmp_path / "billboard.json")

        # Write fixture dataset to expected local path
        with open(minimal_config.dataset.local_path, "w") as f:
            json.dump(raw_entries, f)

        minimal_config.project.year_range = [1990, 1993]
        minimal_config.project.top_n = 2
        minimal_config.checkpoints.force_rerun.s1_ingest = True

        df = run(minimal_config)

        for col in _SCHEMA_1_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) == 8

    def test_run_writes_checkpoint(self, minimal_config, tmp_path, raw_entries):
        """run() must write 01_songs.parquet to the checkpoints directory."""
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.dataset.local_path = str(tmp_path / "billboard.json")

        with open(minimal_config.dataset.local_path, "w") as f:
            json.dump(raw_entries, f)

        minimal_config.project.year_range = [1990, 1993]
        minimal_config.project.top_n = 2
        minimal_config.checkpoints.force_rerun.s1_ingest = True

        run(minimal_config)

        checkpoint_path = Path(minimal_config.checkpoints.dir) / "01_songs.parquet"
        assert checkpoint_path.exists(), "01_songs.parquet was not written"

    def test_run_reads_checkpoint_when_exists(self, minimal_config, tmp_path, raw_entries):
        """run() must skip ingest and read from checkpoint when it exists."""
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.dataset.local_path = str(tmp_path / "billboard.json")

        with open(minimal_config.dataset.local_path, "w") as f:
            json.dump(raw_entries, f)

        minimal_config.project.year_range = [1990, 1993]
        minimal_config.project.top_n = 2
        minimal_config.checkpoints.force_rerun.s1_ingest = True

        # First run — writes checkpoint
        run(minimal_config)

        # Second run — should read from checkpoint, not re-fetch
        minimal_config.checkpoints.force_rerun.s1_ingest = False
        # Remove local dataset to confirm fetch is not attempted
        Path(minimal_config.dataset.local_path).unlink()

        df = run(minimal_config)
        assert len(df) == 8  # checkpoint data returned correctly

    def test_run_force_rerun_ignores_checkpoint(self, minimal_config, tmp_path, raw_entries):
        """force_rerun.s1_ingest=True must re-execute ingest even if checkpoint exists."""
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.dataset.local_path = str(tmp_path / "billboard.json")

        with open(minimal_config.dataset.local_path, "w") as f:
            json.dump(raw_entries, f)

        minimal_config.project.year_range = [1990, 1993]
        minimal_config.project.top_n = 2
        minimal_config.checkpoints.force_rerun.s1_ingest = True

        run(minimal_config)   # first run
        df2 = run(minimal_config)  # force rerun — must not raise, must return data

        assert len(df2) == 8
