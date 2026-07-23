# Final Project: Multimodal Gesture Recognition

End-to-end IoT + data science system: collect real sensor data across
5 sensing modalities, build a machine learning pipeline, and
demonstrate gesture recognition across all 15 gestures.

## Hardware

| Sensor | Board | Role |
|---|---|---|
| IMU | ESP32 / M5Core2 (6-axis) | Wrist-worn motion |
| UWB | 3x DWM3001C | 2 fixed anchors + 1 wrist tag, ranging |
| mmWave | TI IWRL6432FSPEVM | Point-cloud / Doppler, table-mounted near hand |
| WiFi CSI | ESP32-CAM + ESP32-C3 (+ a 3rd board for SoftAP) | Channel state info |
| RFID | SparkFun M7E Hecto (ThingMagic) | Tags on thumb + 2-3 fingers |

## Gestures (15 total)

See `src/gestures.py` for the canonical list: Pull, Push, Clockwise,
Counterclockwise, Left, Right, Bye-bye, One-arm boxing, Clapping,
Two-arm boxing, T-arm, Raise arms, Soli, Open/close fist, Palm up/down.

`gestures.py` also has an `EXPECTED_STRONGEST_SENSOR` table -- our
hypothesis for which sensor should be most useful per gesture. This is
something to **test** in `evaluate.py`, not a restriction: `collect.py`
records every active sensor for every gesture regardless, so one fused
model can be trained across the full gesture set.

## Setup

```
pip install -r requirements.txt --break-system-packages
```

## Pipeline

```
src/collect.py            # record one labeled trial at a time, all active sensors
       |
       v
data/raw/<gesture>/<person>_<trial>_<sensor>.csv
       |
       v
src/combine_datasets.py   # build data/processed/trial_manifest.csv
       |
       v
src/extract_features.py   # -> data/processed/features_{imu,uwb,mmwave,wifi,rfid,fused}.csv
       |
       v
src/train.py               # -> models/{feature_set}_{knn,random_forest}.pkl
       |
       v
src/evaluate.py            # -> results/model_comparison.csv, results/figures/*.png
       |
       v
src/realtime_demo.py       # live prediction using a trained model
```

### 1. Collect data

```
python src/collect.py --person alex --gesture push --duration 2.5 \
    --imu-port /dev/tty.usbserial-AAA \
    --uwb-port /dev/tty.usbserial-BBB \
    --mmwave-port /dev/tty.usbserial-CCC \
    --rfid-port /dev/tty.usbserial-DDD \
    --rfid-tags THUMB_EPC,INDEX_EPC,MIDDLE_EPC,RING_EPC
```

Only pass the ports you actually have connected. Repeat for every
gesture x every group member, aiming for a balanced trial count (20-30
per gesture to start). Start with 4-6 gestures your group can collect
reliably, then expand -- see the lab handout's "practical strategy" note.

### 2. Combine + extract features

```
python src/combine_datasets.py
python src/extract_features.py
```

### 3. Train + evaluate

```
python src/train.py
python src/evaluate.py
```

`evaluate.py` reports, per feature set (imu-only, uwb-only, mmwave-only,
wifi-only, rfid-only, and fused) x model (KNN, Random Forest):
random-split accuracy and held-out-person accuracy, and tells you
whether fusion beat the best single sensor. Confusion matrices land in
`results/figures/`.

### 4. Live demo

```
python src/realtime_demo.py --model models/fused_random_forest.pkl \
    --imu-port /dev/tty.usbserial-AAA --uwb-port /dev/tty.usbserial-BBB
```

## What I need from you to finish two sensor readers

Everything is implemented and real EXCEPT two clearly-marked gaps:

1. **`src/sensors/uwb_reader.py`** -- your DWM3001C's exact shell output
   format is still unconfirmed. Connect one module directly via USB,
   open a serial terminal at 115200 baud, power the anchors, and send
   me what you see -- I'll fill in the real parsing.
2. **`src/sensors/rfid_reader.py`** -- implements the real
   ThingMagic/Mercury binary protocol directly in Python, but I'm not
   100% certain of the exact response byte offsets for your module/
   firmware version. Cross-check against SparkFun's open-source library
   source (`SparkFun_UHF_RFID_Reader.cpp` on GitHub) if reads come back
   garbled, and let me know what you find so I can correct it.

Everything else (`imu_reader.py`, `mmwave_reader.py`, `wifi_reader.py`,
and the entire `collect.py` -> `evaluate.py` pipeline) is complete,
real code.

## Note on data/models in git

`.gitignore` currently excludes raw/processed CSVs and trained `.pkl`
models (they can get large, and shouldn't usually live in git). If
your TA wants the actual dataset/models committed for grading, remove
those lines from `.gitignore` before your first commit.
