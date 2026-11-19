# Domain adaptation plan (webcam data)

Status: **Part 1 (data collection tool) is built. Nothing has been recalibrated or
fine-tuned.** Part 2 starts only after the data below has been collected.

## Why

The liveness model (MobileNetV3-Small, v2) was trained and tested on CelebA-Spoof
dataset photos. On this project's webcam it behaves differently:

- A real face (with glasses, normal room light) often scores P(spoof) a little above
  the UNCERTAIN band's lower edge (`band_low` = 0.0242, from
  `liveness/weights/gray_zone_v2.json`), so the scanner says UNCERTAIN instead of LIVE.
  Real logged values for the owner's face were roughly 0.024–0.17 (`docs/HANDOVER.md`
  Section 7).
- A phone photo was once averaged to P(spoof) 0.0238 and judged LIVE (scan log,
  2026-11-18 11:22 UTC).

That is a domain shift: the camera, compression, lighting, glasses and attack media
differ from the dataset. The fix has to be measured on real webcam data, collected
in a way that keeps a clean, untouched test set.

## Part 1: collecting the data (tool is ready)

**Where:** Admin → Test Mode → "Collection mode (domain adaptation data)".

**What gets saved for each recorded frame (at most one per second):**

| Field | Values |
|---|---|
| Crop | `crop_face()` output (face + 25% margin, before resizing), PNG |
| Label | `real_face`, `real_face_masked`, `printed_photo`, `phone_photo`, `phone_video`, `laptop_screen` |
| Person ID | a code like `P01` (never a name) |
| Condition | `glasses` / `no_glasses`, `normal` / `dim` light |
| Session | `1` (adaptation) or `2` (test only) |
| Scores | P(spoof) of that frame and the 5-frame average, plus the decision at the time |

Files go to `data/domain_adaptation/session_<n>/<label>/` with a `manifest.csv`
(one row per crop). The folder is in `.gitignore`. Each crop also gets a row in the
`test_mode_results` table (`crop_path` column).

Only one face may be in view. Frames with no face or several faces are not saved.

**Consent:** these are real face images of real people. Only collect from people
who agreed, keep the folder off shared drives, and delete it when the work is done.

### How much to collect

**7–8 people, 2 sessions on different days.** The same people in both sessions is
fine because the decision is per-image, not per-identity. If possible, add 1–2
people in session 2 who were not in session 1, so there's also a check on unseen people.

Per person, per session:

| What the camera sees | Conditions | Crops per condition | Crops per person |
|---|---|---|---|
| Real face | glasses / no glasses × normal / dim | 25 | 100 (50 if the person doesn't wear glasses) |
| Real face, masked | normal / dim (glasses as usual) | 15 | 30 |
| Printed photo of a participant's face | normal / dim | 20 | 40 |
| Phone photo (on a phone screen) | normal / dim | 20 | 40 |
| Phone video of a participant | normal / dim | 20 | 40 |
| Laptop screen (photo or video) | normal / dim | 20 | 40 |
| **Total** | | | **240–290** (about 5 minutes of recording at 1 crop/s) |

For 8 people that is about **1,920–2,320 crops per session**, roughly half real and
half attack.

**Why these numbers:**

- **People and conditions matter more than frame count.** Crops taken one second
  apart from the same person in the same minute are very similar, so they are not
  independent samples. 20–25 crops per condition are enough to cover small pose and
  expression changes; more frames of the same minute add little.
- **Every attack is shown in both lights,** because dim light is the condition most
  likely to make screens and real skin look alike to the model.
- **Session 2 must be a different day.** It should also use a different printed copy,
  a different phone if available, and a re-taken photo or video, so session 2 isn't
  just more of session 1.
- **Attack media:** show the attacks at a normal distance, roughly where a real
  face would be, and move them a little as a real person would.

## Part 2: adapting (not started; do this only after both sessions are collected)

**Hard rule: session 2 is never used to choose anything.** No threshold, band,
epoch or setting may be picked by looking at session 2. It is only for the final
before/after report.

### Step 0: baseline on session 2 (before)

Run the current deployed model, unchanged, on every session-2 crop, using the same
preprocessing as the live app (`liveness.predictor.LivenessPredictor`). Report:

- APCER, BPCER, HTER and UNCERTAIN rate, overall, per label, and per light/glasses
  condition.
- Per-person BPCER for the real-face labels, so a few people can't hide behind an
  average.

### Step 1: recalibrate first (no change to model weights)

Using **session 1 only**:

1. Compute P(spoof) for all session-1 crops.
2. Re-select the decision threshold and the UNCERTAIN band with the same method as
   `training/src/evaluate_gray_zone.py`: scan the band width on a fixed grid and
   choose it from target rates, not from test results. Also consider a threshold
   that isn't centered at 0.5, since webcam real faces cluster near 0.02–0.2 rather
   than near 0.
3. Check stability with person-grouped cross-validation inside session 1 (leave one
   person out). Never split one person's crops across folds.
4. Freeze the chosen values, apply them once to session 2, and report before vs. after.

If you approve, the new values replace `liveness/weights/gray_zone_v2.json` and the
`liveness_threshold` setting, recorded as a new file (for example `gray_zone_v3.json`)
with the session-1 data summary. The old file is kept.

### Step 2: fine-tune only if recalibration is not enough

You decide this after seeing Step 1's session-2 results.

- Start from the deployed v2 checkpoint. Fine-tune the head and the last convolution
  block (`feature_extractor.features[12]`) with a low learning rate, keeping the
  earlier layers frozen.
- Training data: session-1 crops mixed with the original CelebA-Spoof face-crop
  training set, so the model doesn't forget the attack types it already handles.
- Validation and early stopping: a person-grouped split of session 1 (held-out
  people), never session 2.
- After training: re-select the band on that session-1 validation split (Step 1
  method), then evaluate once on session 2.
- Also re-run the original CelebA-Spoof test set (n = 3900) and report it next to the
  current 98.69% accuracy / 1.31% HTER, to show whether dataset performance dropped.

### Report format (after Step 1 and, if done, Step 2)

| | Session-2 APCER | Session-2 BPCER | Session-2 HTER | Session-2 UNCERTAIN rate | CelebA test HTER |
|---|---|---|---|---|---|
| Before (v2, current band) | measured | measured | measured | measured | 1.31% (from summary.json) |
| After Step 1 (recalibrated) | measured | measured | measured | measured | re-measured |
| After Step 2 (fine-tuned) | measured | measured | measured | measured | re-measured |

Plus per-label and per-person tables. Every number must come from a saved results
file, as in `training/results/`.

### Known limits of this plan

- 7–8 people is small. Results should be reported as "on these N people", not as
  general accuracy.
- Crops are saved once per second. The live scanner averages 5 frames taken about
  0.15–0.3 s apart, so frame-level results here won't exactly match what the scanner
  shows. Report frame-level results, and treat the scanner's behavior as a separate
  live check (Test Mode, session 2 conditions).
- Browser frames are JPEG (quality 0.85, `static/js/camera.js`), unlike the dataset
  images. This is part of the domain shift and is kept on purpose: the data matches
  what the live system really sees.
