# Future-Tier Features

This directory is a placeholder. It contains no executable code.

All features listed below are formally deferred from the validation-tier
build. They are named here to acknowledge their existence without
introducing any implementation dependency or blocker.

Deferral source: P2 Section 9, P5 Section 13, G3 stage gate.

---

## Deferred features

### Chorus detection
- Depends on: structural lyric parsing (verse/chorus/bridge segmentation)
- Estimated tier: future

### Chorus-specific sentiment and mood analysis
- Depends on: chorus detection
- Estimated tier: future

### Structural contrast score (chorus vs verse delta)
- Depends on: chorus detection
- Estimated tier: future

### Decade cultural mood index
- Depends on: three or more validated decades of sentiment and mood data
- Estimated tier: future

### Cross-decade theme drift index
- Depends on: full dataset with validated themes across all decades
- Estimated tier: future

### Narrative perspective classification (1st/2nd/3rd POV)
- Depends on: additional modeling pass or dedicated LLM call
- Estimated tier: future
- Note: overlaps with subject_focus; adds complexity without additive
  value at validation scale

### Metaphor and symbolic density scoring
- Depends on: fine-tuned classifier or dedicated LLM pass
- Estimated tier: future

### Embedding-based song clustering
- Depends on: semantic_vector field (field slot reserved in Schema 7,
  disabled in validation run per G2-OI3)
- Estimated tier: future

### Subtheme taxonomy
- Depends on: validated primary theme coverage across full dataset
- Estimated tier: future

### Expanded decade aggregation formulas
- Depends on: multiple validated decades
- Estimated tier: future

---

## Activation notes

- `semantic_vector` field slot is reserved in Schema 7 and
  `03_semantic.parquet`. Enable by setting
  `config.theme.semantic_vector_enabled: true` in `config.yaml`.
  The `sentence-transformers/all-MiniLM-L6-v2` model is already
  declared in `config.models.semantic_embedding`.

- All other features require new stage modules, new schema fields,
  and new config sections. No existing validation-tier module needs
  modification to add them.
