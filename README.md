# Final Project - Multimodal Gesture Recognition

For my COSMOS Cluster 10 final project I built a gesture recognition system that uses 5 different sensors (IMU, UWB, mmWave radar, WiFi CSI, and RFID) to recognize 15 different hand/arm gestures. Everything is in Python except for one tiny firmware file that just streams raw IMU data over serial (can't avoid that part since the sensor chip needs something to actually read it).

## The idea

Instead of picking one sensor and hoping it works for every gesture, I used multiple sensors and let the data decide which ones actually matter for which gestures. Some gestures (like a fist closing) are obviously better suited for RFID tags on your fingers, while others (like a full arm swing) make way more sense for the IMU or UWB. So I collect from everything at once and compare single-sensor models against a fused model to see if combining them actually helps.

## Hardware I'm using

- IMU: ESP32 / M5Core2 (6-axis, wrist)
- UWB: 3x DWM3001C (2 anchors + 1 tag on the wrist)
- mmWave: TI IWRL6432FSPEVM, mounted near the hand
- WiFi CSI: ESP32-CAM + ESP32-C3 (from the earlier wifi lab)
- RFID: SparkFun M7E Hecto reader, tags taped to my fingers

## The 15 gestures

Pull, push, clockwise, counterclockwise, left, right, bye-bye, one-arm boxing, clapping, two-arm boxing, T-arm, raise arms, soli, open/close fist, palm up/down. Full list is in `src/gestures.py`.

## How the pipeline works

```
collect.py           -> record one gesture trial at a time, all sensors that are plugged in
combine_datasets.py   -> merges everything into one manifest
extract_features.py   -> turns raw sensor data into ML features per sensor + fused
train.py              -> trains KNN and Random Forest on each feature set
evaluate.py            -> compares them, does held-out-person testing, makes confusion matrices
realtime_demo.py       -> live gesture prediction using whatever model I pick
```

## Setup

```
pip install -r requirements.txt --break-system-packages
```

## How I actually run it

Collecting data (only pass ports for whatever's plugged in that session):
```
python src/collect.py --person sai --gesture push --duration 2.5 \
    --imu-port /dev/tty.usbserial-XXXX \
    --rfid-port /dev/tty.wchusbserial110
```

Then once I've got a bunch of trials collected:
```
python src/combine_datasets.py
python src/extract_features.py
python src/train.py
python src/evaluate.py
```

`evaluate.py` spits out accuracy per sensor and tells me if fusing all the sensors together actually beat the best single sensor, plus confusion matrices land in `results/figures/`.

Live demo once I have a trained model:
```
python src/realtime_demo.py --model models/fused_random_forest.pkl --imu-port /dev/tty.usbserial-XXXX
```

## Debug scripts

Since getting each sensor talking correctly over serial turned out to be its own whole thing, I've got standalone test scripts for when something's not working:
- `src/test_all_sensors.py` - quick check across everything at once
- `src/test_rfid.py` / `src/test_rfid_raw.py` - RFID debugging
- `src/test_uwb_raw.py` - UWB debugging
- `src/test_mmwave_raw.py` - mmWave debugging

## Where I'm at right now

- IMU: working, firmware is in `firmware/imu_streaming/`
- WiFi: working, reuses the sniffer setup from the earlier CSI lab
- mmWave: code's written, still need to confirm the baud rate / port against the real board
- RFID: still debugging - reader connects fine but isn't responding to commands yet
- UWB: still debugging - trying to figure out the right port on the DWM3001C boards (might be plugged into the debug port instead of the actual application port)

## Note on data/models

`.gitignore` currently keeps the raw data and trained models out of git since they can get big - just removing those lines if I end up wanting to commit the actual dataset for grading.
