# Final Project - Multimodal Gesture Recognition

Our COSMOS Cluster 10 final project: a gesture recognition system using 3 working sensors (IMU, mmWave radar, UWB ranging) to recognize 15 gestures. Everything is pure Python except for firmware already flashed by the course labs (no custom firmware needed from us).

## The idea

Instead of one sensor trying to handle every gesture, each gesture is designed around whichever sensor is naturally best suited for it (per our original team plan):

- **IMU (wrist-worn):** pull, push, left, right, bye-bye, one-arm boxing, rest
- **UWB (2 anchors + tag, or wrist-worn pair for bimanual gestures):** clockwise, counterclockwise, clapping, two-arm boxing, T-arms
- **mmWave (table-mounted, close range):** raise arms, soli (thumb), open/close fist, palm up-down

RFID and WiFi CSI were part of the original plan but didn't pan out reliably enough to use (see Status below) — we adapted and leaned harder into the 3 sensors that worked.

## Hardware

| Sensor | Board | Notes |
|---|---|---|
| IMU | ESP32 + BMI270 | Firmware already flashed by the course (IMU_lab_students repo) |
| mmWave | TI IWRL6432FSPEVM | Talks over UART, needs the full CLI config handshake before it streams |
| UWB | 2x DWM3001CDK | Controller + controlee pair, talks Qorvo's UCI protocol |
| RFID | SparkFun M7E Hecto | Built but never got reliable enough to use for real data collection |
| WiFi CSI | ESP32-CAM + ESP32-C3 | Needs a 3rd board (SoftAP) we didn't have -- not integrated |

## How the pipeline works

```
collect.py            -> record one labeled gesture trial, however many sensors are connected
       |
       v
data/raw/<gesture>/<person>_<trial>_<sensor>.csv
       |
       v
combine_datasets.py    -> builds data/processed/trial_manifest.csv
       |
       v
extract_features.py    -> data/processed/features_{imu,uwb,mmwave,fused}.csv
       |
       v
train.py               -> models/{feature_set}_{knn,random_forest}.pkl
evaluate.py             -> results/model_comparison.csv, confusion matrices
       |
       v
augment_features.py     -> synthetic data (Gaussian noise, only used on training folds)
evaluate_augmented.py   -> honest KNN/RF/MLP comparison, with vs without augmentation
train_best_augmented.py -> saves whichever setup actually won as a real model
       |
       v
realtime_demo.py         -> live prediction using ONE model (works, but has a real limitation -- see below)
realtime_router_demo.py  -> live prediction using 3 SPECIALIST models voting (what we actually use)
```

## Setup

```
pip install -r requirements.txt --break-system-packages
```

UWB specifically needs a separate conda environment (`py39`) with the Qorvo UWB tools installed -- see `uwb/` for the cloned `UWB_lab` repo and its own setup instructions.

## Collecting data

```
python src/collect.py --person sai --gesture push --duration 5 --trials 20 \
    --imu-port /dev/cu.wchusbserial-XXXX \
    --mmwave-port /dev/cu.usbserial-XXXX --mmwave-cfg PATH_TO/point_cloud.cfg \
    --uwb-controller-port /dev/cu.usbmodemXXXX \
    --uwb-controlee-port /dev/cu.usbmodemXXXX \
    --uwb-preamble-idx 11
```

Only pass the ports for sensors actually connected that session. `--uwb-preamble-idx` matters a lot -- ours is 11, not the library default of 9; using the wrong one causes the UWB boards to silently fail to range.

## What we actually collected

**Genuine simultaneous 3-sensor data** (all 3 sensors recording together, real fusion test): `push`, `pull` (collected by sai), `clockwise`, `counterclockwise`, `left`, `right`, `bye_bye` (collected by jon)

**Single-sensor data** (collected with just the one intended sensor connected): `rest`, `clapping`, `one_arm_boxing`, `two_arm_boxing`, `t_arm`, `raise_arms`, `soli`, `open_close_fist`, `palm_up_down`

Multiple collectors contributed across gestures: `sai`, `nav`, `jon`, `akshat`, `aks` -- this is what makes the held-out-person evaluation meaningful instead of showing 0/NaN.

## What we found (the honest results)

**Random Forest beat MLP on every sensor, every time**, once we fixed a data-leakage bug in our first augmentation attempt (augmenting before the train/test split let synthetic near-duplicates leak into the test set, making early numbers look artificially good). With a proper leak-free comparison (`evaluate_augmented.py` -- augments only the training fold, after splitting):

- Random Forest wins across imu, uwb, mmwave, and fused feature sets
- MLP underperforms consistently -- our dataset (a few hundred trials per sensor) just isn't big enough for a neural net to beat a tree ensemble
- Data augmentation (Gaussian noise on real feature vectors) genuinely helped mmWave and UWB (our smallest datasets), but didn't help IMU or fused (our biggest datasets) -- makes sense, augmentation helps most where real data is scarcest

**Fusion (all sensors combined) did NOT beat the best single sensor (IMU)** in our overall evaluation. The real reason: most gestures only have one sensor's real data, with the other sensors' feature columns sitting at zero -- the fused model partly learns "which columns are zero" as a shortcut for "which gesture family this is," which works for evaluation-style testing but breaks down in a real live demo where all 3 sensors are actually connected and reporting real (non-zero) values regardless of which gesture is happening.

## Why we use a router, not one fused model, for the live demo

Because of that zero-column shortcut problem, our single fused model performs noticeably worse live than its evaluation numbers would suggest. Instead, `realtime_router_demo.py` loads 3 separate specialist models (one per sensor) and, at each prediction step, asks all 3 "how confident are you, and what's your guess?" -- then reports whichever specialist is most confident. This matches how we actually designed the gesture-to-sensor assignments in the first place, and avoids the zero-padding issue entirely since each specialist only ever sees its own real sensor data.

```
python src/realtime_router_demo.py \
    --imu-port /dev/cu.wchusbserial-XXXX \
    --mmwave-port /dev/cu.usbserial-XXXX --mmwave-cfg PATH_TO/point_cloud.cfg \
    --uwb-controller-port /dev/cu.usbmodemXXXX \
    --uwb-controlee-port /dev/cu.usbmodemXXXX \
    --uwb-preamble-idx 11 \
    --vote-window 5
```

By default this uses the augmented Random Forest models for mmWave/UWB (since augmentation helped those) and the plain real-data model for IMU (since it didn't help there).

## Known issues / limitations (worth being upfront about)

- **UWB is genuinely flaky.** Even after fixing the real bugs we found (wrong default preamble code, restarting the ranging connection every single trial instead of keeping it alive across a session), we still see roughly a 15-50% trial failure rate depending on the day. We over-collect and redo failed trials rather than trusting every attempt.
- **mmWave occasionally gets stuck in a stale CLI state** after heavy use and needs a power-cycle plus a clean config sequence to recover -- not a wiring problem, just a firmware quirk documented in the course's own troubleshooting table.
- **Held-out-person accuracy is still moderate** (roughly 40-55% depending on sensor) -- with only 2-5 collectors, the model hasn't seen enough variation in how different people perform each gesture to generalize as well as the random-split numbers suggest.
- **RFID and WiFi CSI were never integrated** into the final data collection -- RFID's protocol implementation never got fully verified against the actual hardware, and WiFi CSI needs a 3rd board (SoftAP) we didn't have available.

## Note on data/models in git

`.gitignore` no longer excludes collected data or figures -- everything is tracked so the full dataset and trained models are part of the repo history, not just the code.
