# Raw temporal imputation experiment v2

This is an exploratory infant raw-BOLD reconstruction experiment. Movie decoding, gaze-conditioned analyses, verified cortical gray-matter signals, and recovery of the true signal during actual motion are outside its present scope.

## Experimental unit and data

Deduplicate by SHA256 of the full uncompressed raw acquisition. Preserve movie-folder aliases as provenance, never as separate training examples. Reconsider all 64 unique acquisitions in the 72-entry source inventory instead of retaining v1 exclusions. The frozen v2 protocol includes 53 acquisitions and excludes 11 for insufficient eligible observations. Original family partitions are preserved: development 39 acquisitions / 18 families; reserved test 14 acquisitions / 5 families. Five development folds rotate validation families while preserving test families. Fold 0 has 32 training, 7 validation and 14 reserved test acquisitions.

Raw times remain on the original TR grid. Actual rejected samples are NaN and never loss/evaluation targets. Artificial targets come only from QC-eligible values; they are removed from model input before normalization. Predictions at actual rejected samples are labeled imputed with observed-context distances and have no ground-truth accuracy score.

## Calibration, anatomy and frame QC

Frames 0–10 are reserved and never scored. Choose a reference among frames 3–10 using native-slice quality, without reading analysis frames. This experimental prefix is not a reconstruction of original burn-in metadata. The published functional-to-anatomical affine is only an initializer. Correct its head pose by registering the chosen reference to the anatomical image with a six-DOF residual and Mattes mutual information. No released functional centroid or processed/interpolated BOLD amplitudes are used.

Each analysis volume is registered independently to the reserved reference. Select at most 256 spatially spread voxels from the intersection of eroded occipital ROI, eroded anatomical brain support and reference intensity support. There is spatial linear resampling, but no temporal interpolation, filtering, despiking, slice-time correction, temporal warm start or mean-BOLD reference. The ROI includes brain tissue beyond verified cortical gray matter.

Existing declared RMS motion, fractional DVARS, correlation and coverage thresholds are retained. Also assess each original volume in its native slice grid with the motion-mapped calibration brain mask. Reject alternation residual greater than 0.04, 90th-percentile absolute slice residual greater than 0.35, or insufficient slice support. These are frozen engineering screening rules with synthetic artifact tests, not validated clinical infant cutoffs. MRIQC-style temporal z-normalization is not used to filter the BOLD signal.

The first analysis frame is excluded from loss/recovery because this representation does not calculate its adjacent predecessor. It does not treat a nonadjacent chosen reference as the previous TR. Recoverable slots include motion and other QC rejections only with finite registration and field-of-view coverage. Failed registration does not silently become an aligned imputation target.

Independence from hidden values is conditional on frozen QC and the external initial affine, whose original estimation may have used analysis frames. MI improvement and brain-mask inclusion do not certify anatomical accuracy. AI-assisted inspection of three orthogonal review planes checks gross placement; expert segmentation review and QC sensitivity analyses remain necessary before strong neuroscientific claims.

## Training and evaluation

- STSR starts from random initialization; pretrained/resumed weights are refused.
- Sample a family uniformly, then an acquisition within it. Log insufficient-context retries.
- Fit irregular corruption-mask patterns on training families only and exclude the current family's donor pattern.
- Use fixed evaluation masks shared across methods, independent of initialization seed.
- Normalize each voxel using observed samples after artificial masking.
- Compare STSR, linear interpolation and adjacent-included-value interpolation on identical targets.
- Select the checkpoint by normalized MSE averaged across masks, acquisitions within each family, then equally across validation families. Keep pooled scores as secondary diagnostics.
- Report gap duration and interior/leading/trailing support. The 16/32-second stress cases are separate from checkpoint selection; retain unsupported cases.
- Do not access reserved test prediction metrics during selection or tune QC thresholds from test results.

The first question is whether irregular-mask training improves held-out raw-BOLD reconstruction relative to temporal interpolation, including long and one-sided gaps. A future uniform-mask ablation must match masking burden and use identical evaluation masks; this change does not claim that ablation has been run. Movie decoding needs separately verified timing and gaze mapping.

## Verification and provenance

Preparation job 61036981 completed all 64 acquisitions. The separate reconstruction audit checked 1,467,648 included values, motion/DVARS/correlation/mask calculations and 546 training/validation mask draws. All included points lie inside eroded anatomical brain support. Native-slice labels are independently recomputed from original ZIPs before release. A real-record gradient check verifies execution, not model efficacy.

The frozen preparation code and training code have separate hashes: family weighting was finalized after preprocessing and does not change data. QA approval is bound to each manifest, input NPZ and unique raw hash. V1 data and its cancelled checkpoint remain blocked.

Published dataset: `/pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3/derived/raw_qc_v2_20261109/`.

The geometry follows [FSL FLIRT conventions](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/faq.html); calibration uses [SimpleITK registration](https://simpleitk.readthedocs.io/en/main/registrationOverview.html). Slice thresholds above are our declared engineering protocol, not MRIQC IQMs or established infant clinical criteria.
