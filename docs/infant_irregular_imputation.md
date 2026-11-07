# Infant movie fMRI: irregular temporal imputation pilot

This independent project reuses `SpaceTimeAttentionNet.features()` and its INR readout to
recover missing **BOLD samples on the native acquisition grid**. The first
experiment is temporal imputation. There is no spatial upsampling, temporal
decimation, HRF inversion, or claim of measured sub-TR resolution in this path.

The acquisition lattice remains uniformly spaced. Its **observation mask can
be irregular**: isolated frames, several bursts, leading/trailing gaps, and
voxel-specific invalidity. Removing frames and closing up the time axis would
change the problem and is rejected. Factorized temporal attention spans the
entire supplied run/ROI (`factorized_time_window: 0`). This is offline imputation
using earlier and later context, not causal prediction.

## Learning and inference

For a clean/usable mask `valid`, draw a synthetic temporal holdout `H`:

```
condition = valid & ~H
target    = valid & H
```

The default `fitted` training sampler fits gap templates only on training
families. It samples a donor family, then a nonempty-gap run, preserves that
run's gap durations/count and leading/trailing status, and randomizes interior
positions without overlap. Sampling durations in seconds also supports a
different TR. The recipient's family is excluded from donor selection. At least
two nonempty donor families are therefore needed for training. Empty donors
cannot supply a supervised target; impossible layouts are rejected rather than
silently shortened. `mask_audit.jsonl` records the chosen donor, attempts,
proposed gaps, realized targets, and skipped training records.

The `legacy` ablation retains the previous mixture of random bursts and observed
training patterns. Set `mask.empirical_probability: 0` for random-only training,
or `0.5` for the previous mixture. The legacy length list/max-gap settings do
not govern the fitted sampler. The mask generator requires at least two remaining
observations in each scored voxel. Actual motion, gaze exclusions, padding,
and unknown values are never targets. Masking happens before observation-only
normalization and STSR feature extraction. Stored values at invalid locations
are discarded, including interpolation already present in the public data.
Only target queries contribute to the normalized MSE loss. Both trunk and INR
are trained from scratch; existing SR checkpoint weights are not loaded.

Inference copies all valid observations and fills only `recoverable` slots with
at least two observations in the same voxel. Other excluded slots stay NaN.
Outputs include `valid`, `imputed`, `recoverable`, left/right context distances
in seconds, and a two-sided-context flag. These are support diagnostics, not
calibrated uncertainty. Long and boundary gaps may still be poorly recoverable.
The query API also supports in-range non-native times, but those estimates have
no measured fine-time target in this dataset.

## Data contract and splits

A JSON manifest lists prepared records:

```json
[
  {"id": "movie:participant_session", "family": "family_id", "split": "train", "path": "record.npz"}
]
```

Use distinct families for `train`, `val`, and `test`, keeping repeats, siblings,
and parent/child records together if adults are later added. Assign families
before preparing patches or sampling masks. The manifest rejects family overlap,
duplicate IDs, and duplicate resolved paths. Evaluation also checks identities
against the checkpoint's training/model-selection manifest. Family labels must
be curated correctly; the code cannot discover unrecorded family relationships.
Subject generalization is separate from movie generalization: reserve movie
identities as an additional held-out experiment when testing the latter.

Each NPZ contains:

| Field | Shape/type | Meaning |
|---|---|---|
| `values` | V × T float32 | Native-grid signal, without new full-run normalization |
| `coords_mm` | V × 3 float32 | Physical coordinates in the supplied run's space |
| `times_s` | T float32 | Original uniform time grid, seconds |
| `valid` | V × T bool | Usable observed samples eligible for artificial holdout |
| `recoverable` | V × T bool | Actual motion gaps eligible for inference; disjoint from valid |
| `padding` | T bool | Optional protocol-excluded tail; absent means all frames are in the analysis domain |

The legacy field name `padding` denotes a protocol-excluded tail. EDA found
nonzero values beyond CSV `num_TR` in three inspected runs; these slots are not
necessarily zero-filled or absent from acquisition. The current protocol
conservatively excludes them pending raw-to-movie alignment. The code's
`acquired`/`acquired_frames` terminology refers to this selected analysis prefix,
not proof of the raw acquisition interval.

This excluded tail must be contiguous and cannot contain valid/recoverable samples.
It does not contribute to gap statistics or stress-test duration. `Child_Play`
is currently rejected by the adapter and manifest reader: the dataset README
states that it concatenates four clips after removing rest intervals. Its
stored regular array is not a continuous acquisition timeline. It needs an
explicit original-time/clip-boundary mapping and separately prepared continuous
segments before use here.

This initial runner uses full runs with at most 256 frames and a predefined ROI
or patch of at most 256 voxels. It does not stitch a whole-brain reconstruction.
Select geometry from anatomy or an independently defined atlas, never from
held-out BOLD performance. Cross-subject anatomical correspondence must be
established separately when native-space ROIs are compared.

## Prepare the downloaded Dryad data

The adapter reads selected members of the existing ZIPs, retaining original
files. It streams one volume at a time, samples explicit voxel coordinates,
uses the native affine/TR, and supports the wrapped member offsets found in
some source archives. CRC checks cover members actually read, not every archive.

Create a selection JSON with one entry per infant run, and an NPY integer
`V × 3` voxel-index array for each ROI. Paths to index arrays are relative to the
selection file. Use the actual movie/participant IDs in the downloaded CSVs:

```json
[
  {"movie": "Meerkats", "ppt": "ACTUAL_PARTICIPANT_ID", "family": "CURATED_FAMILY_ID",
   "split": "train", "voxel_ijk": "roi_voxels.npy"}
]
```

```bash
python -m stsr.imputation.prepare_dryad \
  --selection selection.json \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --eye-shift-tr 0 --guard-tr 1 \
  --out outputs/infant_prepared
```

`--eye-shift-tr` is required as an explicit protocol choice. Zero above means
use the stored gaze confound alignment; it is not a validated universal gaze
lag. Do not automatically transfer one paper's gaze lag to every movie. A
positive shift delays the gaze exclusion without circular wrapping and marks
the newly unknown leading slots excluded. `--guard-tr` additionally excludes
neighbors of bad frames from observations/targets; a fixed guard does not undo
nonlocal temporal filtering. The adapter requires finite gaze metadata, uses
CSV `num_TR` to mark the unobserved tail, and excludes motion/gaze overlap from
recoverable motion. Review `preparation.json` and mask arrays before training.

The source native preprocessed series already underwent preprocessing,
potentially including motion interpolation and temporal filtering. Discarding
marked values now cannot remove their influence on neighboring values. Thus
this adapter supports a **preprocessed-data wiring/feasibility pilot**, not
proof of leakage-free reconstruction from raw fMRI. A scientific benchmark
must track raw-to-movie time alignment and censoring, and either avoid temporal
mixing before synthetic holdout or reprocess with holdout-aware operations.
Raw-run masks cannot simply be copied from these processed arrays.

## Run and assess

Install this package using the independent repository's README. Its local
`.venv/bin/python` and installed `infant-impute` entry point use this repository's
source, without importing the original fmri_stsr checkout.
Output directories must be new to prevent accidental replacement.

```bash
python -m stsr.imputation.runner train \
  --manifest outputs/infant_prepared/manifest.json \
  --config configs/infant_irregular_imputation.yaml \
  --device cuda --out outputs/infant_irregular_pilot

python -m stsr.imputation.runner evaluate \
  --manifest outputs/infant_prepared/manifest.json \
  --checkpoint outputs/infant_irregular_pilot/best.pt \
  --split test --device cuda --out outputs/infant_irregular_pilot/test.json

python -m stsr.imputation.runner infer \
  --record outputs/infant_prepared/SELECTED_RECORD.npz \
  --checkpoint outputs/infant_irregular_pilot/best.pt \
  --device cuda --out outputs/infant_irregular_pilot/imputed.npz
```

Training requires both train and validation records. The default
`evaluation_mask_strategy: fitted` selects fixed validation masks from the
saved **training** profile, regardless of the training masking strategy. Keep
this setting and split/seed fixed when comparing random-only, legacy mixture,
and fitted training. There are ten primary masks per validation run by default.
Checkpoint selection uses only this primary validation MSE. Test signals are
not loaded during training. Provenance
includes split membership, configuration, file/source hashes, base revision,
and dirty-tree status, plus the complete fitted profile. The profile is restored
with the checkpoint and is never refitted on evaluation data. The runner saves the best model for inference, not an
optimizer/RNG resume checkpoint.

STSR, linear interpolation, and mean-of-bracketing-observations interpolation
are evaluated on exactly the same artificially hidden values. At boundaries,
the baselines use the nearest available observation. Reports include matched
counts, MSE/MAE, per-record mean voxel correlation with its contributing-row
count, and two-sided/leading/trailing errors. `bracket_*` strata measure the
span between bracketing observations. `missing_*` strata measure the duration
of the unobserved interval containing a target, including existing exclusions
but excluding padding; they also report one-sided long gaps.

`stress_evaluation.seconds: [16, 32]` separately tests a single interior,
leading, and trailing gap at each duration. Only known values inside that gap
are scored, and the report includes the scored fraction. Cases without adequate
context/targets are reported as unsupported with zero scored count. Stress
scores are never pooled into the primary score or checkpoint selection.
Synthetic gap evaluation does not provide ground truth for real motion gaps;
motion can be associated with behavior and differ from artificial masking.

The initial research questions are:

1. Does irregular-mask STSR improve native-TR BOLD reconstruction over the two
   interpolation baselines on held-out families, across gap durations/support?
2. Does adding training-family empirical motion patterns improve transfer to
   independently held-out missingness patterns versus random bursts alone?
   The fitted and legacy/random-only training strategies now support this
   comparison using the same fitted validation masks; trained comparisons
   remain to be run.
3. Under a fixed movie decoder trained independently of test masks, does recovery
   improve scene/event identification or time-segment retrieval? Compare the
   censored input, interpolation, STSR, and artificially unmasked reference with
   the same masks and decoder. No movie decoder is implemented in this pilot.

Use family-level resampling and several seeds for scientific comparisons;
repeated masks on one run are not independent subjects. The primary test masks
come from training-family templates. Independent held-out-motion-pattern and
movie-decoding evaluations remain separate experiments.

## Dataset mask audit and calibration (2026-11-06)

The local audit read metadata, confounds, and native NIfTI headers for all
94 infant movie records across five movies; no BOLD arrays were read. Family
IDs were taken from the dataset README's explicit `sXXXX` definition. Before
reading any masks, 35 families were assigned by the SHA256 rank of
`42:family_id` to floor(70%) train, floor(15%) validation, and the remainder test.
The frozen assignments are saved in `cohort_split.json`.

The primary protocol here uses the stored gaze alignment (`eye_shift_tr=0`),
no added guard (`guard_tr=0`), and the first CSV `num_TR` frames as the acquired
interval. These are explicit pilot conventions, not a validated gaze/HRF lag
or proof of raw-data temporal independence. Source confound/CSV hashes and
native header hashes accompany the report. `Child_Play` receives separate
stored-axis descriptions but is excluded from continuous-time calibration.
Records with unknown gaze cannot contribute clean targets. The resulting
eligible split is **46 train records / 19 families, 6 validation records / 2
families, and 19 test records / 5 families**. The small validation-family count
limits statistical precision; this pilot split is not a substitute for
prespecified family-level cross-validation in a scientific comparison.

All sampler fitting/comparison below uses only the 46 eligible training records:

| Gap definition/method | Median gap | 95th percentile | Maximum gap | Boundary-gap fraction |
|---|---:|---:|---:|---:|
| Motion alone, before gaze exclusion | 4 s | 12 s | 16 s | 8.45% |
| Actual recoverable motion, after gaze exclusion | 2 s | 6 s | 16 s | 4.35% |
| Previous random/empirical mixture: realized clean targets | 4 s | 28 s | 64 s | 3.32% |
| Fitted sampler: realized clean targets | 2 s | 6 s | 16 s | 3.29% |

Recoverable motion comprised 115 gaps / 181 frames in 4,101 acquired training
frames. Of these gaps, 75 were one TR, 24 two TRs, and 12 three TRs. Nineteen
training records had no recoverable motion; the maximum gap count per record
was 13. The five boundary gaps were all leading gaps. This motivates separating
rare/unsupported long and trailing conditions into stress evaluation, rather
than treating the old uniform length menu as the natural task distribution.

The comparison used 100 draws per training record per method (4,600 each),
sampling on the full stored axis with false padding and measuring within the
acquired prefix. All requested draws supplied targets. Fitted sampling needed
1.073 attempts on average, at most four. Its mean hidden fraction of *clean*
input was 6.96%, versus 11.59% for the previous mixture. Those fractions use a
different denominator from actual motion frames divided by acquired frames.

Matching selected quantiles does not establish exact distribution matching.
The fitted sampler conditions on nonempty donors, balances donor families,
randomizes interior positions, and intersects proposals with recipient validity.
These operations change frequencies, spacing, and edge incidence; both proposed
and realized distributions are recorded. In particular the realized boundary
fraction remains below the run-pooled real fraction. No optimizer, learning-rate,
model-size, or 1,000-step tuning claim follows from this mask-only audit.

```bash
python -m stsr.imputation.audit_masks \
  --data-root /pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3 \
  --eye-shift-tr 0 --guard-tr 0 --draws 100 \
  --out outputs/infant_motion_audit_2026-11-06_final

python -m stsr.imputation.plot_mask_audit \
  --audit outputs/infant_motion_audit_2026-11-06_final/audit.json \
  --out outputs/infant_motion_audit_2026-11-06_final
```

Use a fresh output directory for a rerun. The final directory contains
`audit.json`, `records.json`, `cohort_split.json`, `train_mask_profile.json`,
and PNG/PDF comparison plots. Reuse the frozen family assignments in the
ROI preparation selection. The trainer fits its profile from the actual
prepared training records, so the profile can differ if that selection changes.
The earlier directory without `_final` used cropped-axis sampler simulation;
the `_final` result above supersedes that comparison.

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest tests/test_irregular_imputation.py -q
```

Tests cover invalid-value invariance, zero hidden-input gradients, clean-target
loss, trunk/INR gradients, observed-value preservation, physical-time queries,
family separation, baseline denominators, gaze/padding exclusions, streamed
NIfTI geometry, and training/checkpoint/reload/inference. Small synthetic tests
verify execution and isolation; they do not establish infant decoding gains.

Historical upstream verification: 20 tests passed when running this file together with
`tests/test_stsr_design.py`. A read-only-source CPU smoke used Meerkats
`s0607_1_4`: 8 fixed probe voxels × 93 stored frames at TR 2 s, with 504 valid
values and 112 motion-only recoverable values after gaze/tail exclusion.
Forward/backward produced finite loss/gradients; inference preserved all valid
values and left the final three excluded frames NaN. The probe used random
weights and an arbitrary fixed cube, so it is not a trained-model result or
an anatomically defined visual ROI analysis. Local receipts are under
`outputs/infant_irregular_wiring_smoke/` (ignored by Git).

In the original repository, after the mask-calibration extension, **29 tests passed** with
`python -m pytest tests/test_irregular_imputation.py tests/test_stsr_design.py -q`.
Additional coverage includes training-only profile fitting, family exclusion,
gap-duration preservation across TRs, padding-aware edge metrics, incompatible
Child_Play rejection, matched evaluation across training strategies, separate
unsupported stress cases, and checkpoint/profile restoration. These tests and
the mask-distribution audit do not constitute infant model training or a
measured BOLD/decoding performance improvement.

The independent repository carries the 20 imputation tests and five additional
EDA tests: **25 tests passed** with `python -m pytest -q`. The upstream
`test_stsr_design.py` covers broader original-project paths and is not included
here; the imported attention/INR path is exercised by the retained end-to-end tests.
