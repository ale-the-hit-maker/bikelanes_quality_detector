# Bikelanes_quality_detector
Project Assignment of Data Analysis &amp; Learning Methods course, from Technology University of Eindhoven

**Group41**: Alessandro Villani, Alessandro Marcellini, Manuel Cardia
 
We use the accelerometer, gyroscope and gravity sensor of a smartphone mounted on a bike handlebar to decide whether a stretch of bike lane is **smooth** or **bumpy**. This repository contains our recordings, the full analysis notebook (preprocessing → feature engineering → feature selection → modeling → evaluation → deployment on the external dataset) and the report.
 
 
## Contents of the submission
 
```
group41_submission.zip
├── group41.ipynb          # the whole pipeline, executed, no external .py files
├── group41_Report.pdf     # CRISP-DM report (max. 8 pages)
├── README.md             # this file
└── data/                 # raw sensor recordings (one folder per recording)
    ├── bumpy_<description>[-<timestamp>]/
    └── smooth_<description>[-<timestamp>]/
```
 
Each recording folder is the unzipped export of the *Sensor Logger* app and contains:
 
| File | Content |
| --- | --- |
| `Accelerometer.csv` | linear acceleration, m/s², columns `time`, `seconds_elapsed`, `x`, `y`, `z` |
| `Gyroscope.csv` | angular velocity, rad/s, same columns |
| `Gravity.csv` | gravity vector in the phone axes, m/s², same columns |
| `Metadata.csv` | platform (iOS / Android), standardisation flag, … (used to fix the sign convention) |
 
> **Before running:** the notebook reads the recordings from `RAW_REC_DIR` (first code cell) and writes intermediate results to `PRE_PROCESSING_DIR`. Both are defined at the top of the notebook as relative paths. Set `RAW_REC_DIR = "data"` (or wherever you unzipped the data). The intermediate folders are created automatically and can be deleted at any time.
 
---
 
## Setup
 
**Python 3.12 or newer** is required (the notebook uses f-strings with nested quotes, which are not valid in older versions).
 
Only the packages used in the instruction sessions are needed:
 
| Package | Version used |
| --- | --- |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |


 
 
 
All randomness is seeded (`random_state=42`) and the cross-validation splits are deterministic (`GroupKFold`), so the numbers should be reproduced exactly on the same package versions. Expect the feature-selection cell to be the slowest one, because it runs a forward selection with an inner cross-validation for four models in each of the five outer folds. It also prints a long log (mutual-information ranking and dropped features per model and fold); this is intentional and is the reason the executed notebook is several MB.
 
---
 
## Dataset description
 
**Task.** Binary classification of short time windows: `0 = smooth`, `1 = bumpy`.
 
**Sensors and settings.** *Sensor Logger* (iOS/Android), only Accelerometer, Gyroscope and Gravity enabled, default 100 Hz, *Standardisation* option on.
 
**Phone placement.** Fixed vertically on the straight part of the handlebar, screen facing the rider, with a bike phone support. We did not use a pocket or a body mount on purpose: a rigid handlebar mount keeps the rider's legs and the movement of the phone out of the signal, so what remains is the bike on the road. The tilt of the holder is not identical on every bike, so the notebook rotates every recording into a common *bike frame* using the gravity vector.
 
| Axis | Direction in the reference position |
| --- | --- |
| x | along the handlebar, positive towards the rider's left |
| y | perpendicular to the road, positive downwards |
| z | horizontal, positive towards the direction of travel |
 
**Recordings.** 15 recordings: 9 smooth, 6 bumpy, collected in and around Eindhoven in September 2026 by 2 riders on 2 different phones. A *smooth* lane is a regular red cycling lane without visible imperfections; a *bumpy* lane has noticeable roughness (potholes, cracks, uneven surfaces) but is still safe and representative of lanes that people use. The label is **one per recording** and is encoded in the folder name (`smooth_…`, `bumpy_…`), which is how the notebook reads it. Total duration: `TBD`. Final number of windows: 630 (`TBD` smooth / `TBD` bumpy).
 
| Class | Recording folders |
| --- | --- |
| bumpy (6) | `bumpy_con_tante_buche`, `bumpy_con_una_grande_curva_alla_fine`, `bumpy_stadio`, `bumpy_stazione`, `bumpy_stradabumpy_con_rialzi_e_buche`, `bumpy_swapfiets` |
| smooth (9) | `smooth_con_fermata`, `smooth_con_fermata_2`, `smooth_con_un_po_di_rialzi_e_piccoli_tratti_saltellanti`, `smooth_grande_curva_apl_inizio`, `smooth_ma_con_irregolarita_periodiche_foto_`, `smooth_monknatlab`, `smooth_quartiere_bene_eindhoven`, `smooth_strada_universiya_ingresso`, `smooth_strda_turca` |
 
(Some folders also carry a timestamp suffix such as `-2026-09-21_13-24-26`.)
 
**Known limitations of the data.** Small dataset; one label per recording; bike, phone, rider, session and speed are not controlled, so they can be partly confounded with the class; no GPS, so slope, yaw and speed are not corrected. These are discussed in the report.
 
---
 
## What the notebook does
 
1. **Preprocessing**
   1. Sign convention: everything converted to the iOS convention (metadata first, gravity sign as fallback).
   2. Trim the first and last 1.5 s of each recording (screen taps).
   3. Remove stationary segments (rolling energy of the accelerometer magnitude below a threshold for at least 2 s).
   4. Axis alignment: one constant rotation per recording (`Rz(ψ)` then `Rx(θ)`, from the median gravity) into the common bike frame, with automatic checks printed per recording.
2. **Exploratory spectral analysis:** whole-signal FFT, Welch PSD (bumpy vs. smooth), spectrograms, used to design the filter.
3. **Band-pass filter:** 4th-order Butterworth, zero phase, accelerometer 10–40 Hz, gyroscope 5–40 Hz, applied separately to each continuous segment.
4. **Windowing and labelling:** 3 s windows, 30 % overlap, never across a removed segment; the label comes from the recording name.
5. **Feature extraction:** 105 time- and frequency-domain features per window (17 per accelerometer channel, 18 per gyroscope channel).
6. **Feature selection** (inside each outer training fold)
   - *Supervised models:* mutual information ranking → correlation filter (|ρ| > 0.8) → model-specific forward selection with an inner grouped CV (4 to 12 features).
   - *Unsupervised models:* correlation filter (|ρ| > 0.9) → standardisation → PCA (max. 6 components). No labels are used.
7. **Normalisation:** `StandardScaler` fitted on the training part of each fold only.
8. **Modeling and evaluation:** 5-fold `GroupKFold` with the recording as group.
   - Supervised: KNN, Logistic Regression, Random Forest, SVM (RBF).
   - Unsupervised: K-Means, Hierarchical clustering, GMM, DBSCAN (on the PCA components).
   - Metrics: macro-F1, accuracy, per-class recall, confusion matrices; ARI for the unsupervised models. ` TODO : update with the final list`
9. **Deployment:** the selected model is applied to the independent external dataset (never used for training or testing) and the smooth and bumpy sections are visualised.

> **Leakage.** Windows from the same recording are never split between training and test (they overlap and are almost identical). Feature selection, scaling and PCA see only the training recordings of each fold. One exception that we know of: the filter band was chosen by inspecting the spectra of all recordings; it is a wide band and the choice was visual, but it is not completely independent of the test folds.
 
