# EnsembleAI Hackathon 2026 — ChEBI Ontology Prediction Plan

## Task Summary

**Goal:** Predict ChEBI ontology terms for small molecules.
**Input:** SMILES strings.
**Output:** Binary predictions for 500 classes per molecule.
**Structure:** Classes form a DAG (directed acyclic graph) — if a child class is predicted, its parent must also be predicted.
**Metric:** Macro-averaged F1 score. Tiebreaker (<1% difference): average number of DAG inconsistencies (child predicted with higher probability than parent).
**Data:** `chebi_dataset_train.parquet` (SMILES + IDs + 500 binary labels), `chebi_classes.obo` (DAG structure), `chebi_dataset_test_empty.parquet` (molecules to predict).

## Key Considerations

- **Macro F1** means every class counts equally — rare child classes matter as much as common parent classes. This heavily penalizes models that ignore minority classes.
- **DAG consistency** is a tiebreaker but also likely correlated with F1 — enforcing it should improve predictions, not just satisfy rules.
- **500 binary tasks** — this is massively multitask, similar to ToxCast (617 tasks) in MoleculeNet. MOLTOP paper shows RF handles this, though it's the slowest part.
- **Imbalanced classes** — child nodes will be rare. Need per-class threshold tuning or class weighting.

## Plan

### Phase 1 — Baseline (first ~2 hours)

**1a. Data exploration**
- Load train parquet, inspect label distribution across the 500 classes.
- Parse the `.obo` file to extract the DAG (parent-child relationships).
- Check class imbalance: how many positives per class? How deep is the hierarchy?
- Verify train/test molecule overlap (there should be none).

**1b. ECFP + Random Forest baseline**
```
scikit-fingerprints pipeline:
  MolFromSmilesTransformer → ECFPFingerprint(count=True) → RandomForestClassifier
```
- RF natively supports multi-output classification.
- Use `class_weight="balanced"` or `"balanced_subsample"` to handle imbalance.
- Submit this immediately to get on the leaderboard.

**1c. MOLTOP baseline**
- Use the open-source MOLTOP code to extract features.
- Same RF classifier setup.
- Compare to ECFP — the MOLTOP paper shows these are competitive; one may dominate the other depending on how structural vs. functional the ChEBI classes are.

### Phase 2 — Improve featurization (~2 hours)

**2a. Feature stacking**
- Concatenate multiple fingerprints: ECFP (count) + MACCS + Atom Pair + Topological Torsion + RDKit 2D descriptors.
- Use `make_union` from scikit-learn with scikit-fingerprints transformers.
- Substructure fingerprints (MACCS, Laggner, Klekota-Roth) are especially relevant here since ChEBI classes often correspond to structural motifs and functional groups.

**2b. Classifier upgrade**
- Switch from RF to XGBoost/LightGBM with per-class binary classifiers (One-vs-Rest).
- LightGBM handles high-dimensional sparse features well and is faster than RF for 500 tasks.
- Tune decision thresholds per class on a validation fold (optimize F1 per class, not accuracy).

**2c. Per-class threshold tuning**
- For each of the 500 classes, sweep thresholds on validation set to maximize F1.
- This is critical for macro F1 — default 0.5 threshold is suboptimal for imbalanced classes.

### Phase 3 — DAG consistency enforcement (~1 hour)

**3a. Parse the DAG from `.obo` file**
- Build adjacency structure: for each class, know its parents and children.
- OBO format: look for `[Term]` blocks, `id:`, `name:`, `is_a:` relationships.

**3b. Post-hoc consistency correction**
- Work with predicted probabilities (not binary predictions).
- **Bottom-up propagation:** for each parent, set `P(parent) = max(P(parent), max(P(children)))`. This ensures no child has higher probability than its parent.
- **Top-down pruning (optional):** if a parent probability is below threshold, force all descendants below threshold too.
- Apply per-class thresholds AFTER consistency correction.

**3c. Consistency-aware thresholding**
- After propagation, apply thresholds.
- Verify zero inconsistencies in final output.

### Phase 4 — Advanced methods (if time permits)

**4a. Hierarchy-aware training**
- Instead of flat multi-output, train per-level classifiers.
- Level 0 (roots): broad categories.
- Level 1+: classifiers conditioned on parent predictions, using only samples where parent is positive.
- This focuses each classifier on the relevant subset and can improve rare child class predictions.

**4b. Neural model (if dataset is large enough)**
- Simple feedforward network on concatenated fingerprint features with a hierarchical loss:
  `L = BCE_loss + λ * Σ max(0, P(child) - P(parent))` for all parent-child pairs.
- This learns consistency during training rather than enforcing it post-hoc.

**4c. Ensemble**
- Blend predictions from multiple featurizations/classifiers.
- Simple averaging of probabilities from RF(ECFP), RF(MOLTOP), LightGBM(stacked fingerprints).
- Apply DAG consistency correction on the ensembled probabilities.

**4d. SMARTS pattern rules**
- For well-defined structural classes (e.g., "Aromatic compound", "Carboxylic acid"), write direct SMARTS pattern matches using RDKit.
- These give perfect precision for classes that are purely structural.
- Use as hard overrides on top of ML predictions for high-confidence classes.

### Phase 5 — Final submission

- Select best model(s) based on cross-validation macro F1.
- Apply DAG consistency correction.
- Apply per-class optimal thresholds.
- Verify output format matches `chebi_submission_example.parquet` (correct columns, IDs).
- Verify zero inconsistencies.
- Push code to public GitHub repo.

## What to Skip

- **Pretraining-heavy models** (GROVER, GraphMVP, GEM) — setup time too high, gains unreliable.
- **Conformer generation / 3D fingerprints** — slow, not needed for ChEBI structural/functional classes.
- **GNNs from scratch** — unless dataset is very large; fingerprints are likely stronger given time constraints.
- **ChemBERTa/SMILES transformers** — possible but tuning for 500-class multi-label is tricky in limited time.

## Key Libraries

- `scikit-fingerprints` — fingerprint computation, molecular preprocessing, scaffold splits
- `scikit-learn` — RF, pipelines, feature unions, multi-output classification
- `xgboost` / `lightgbm` — gradient boosted trees
- `rdkit` — SMILES parsing, SMARTS matching, molecular utilities
- `networkx` or manual parsing — DAG structure from `.obo` file
- `pandas` / `pyarrow` — parquet I/O

## Execution Priority

1. **Get a submission on the board** → ECFP + RF, no hierarchy, default thresholds
2. **Threshold tuning** → big macro F1 gain for free
3. **DAG consistency** → fixes tiebreaker, may also improve F1
4. **Better features** → stacked fingerprints, substructure FPs
5. **Better classifier** → LightGBM
6. **Ensemble + SMARTS rules** → final push