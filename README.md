# Final Project: Gesture Recognition with IMU + UWB Fusion

## Hardware Setup

| Role | Device | Placement |
|---|---|---|
| Primary sensor | Core2 IMU | Strapped to wrist (accelerometer + gyroscope) |
| Anchor 1 | UWB module #1 | Fixed, placed in front of the gesture space |
| Anchor 2 | UWB module #2 | Fixed, placed to the side of the gesture space |
| Tag | UWB module #3 | Strapped to the *same* wrist as the IMU |

Both the IMU and the UWB tag connect to your laptop over USB at the same
time (use the 4-port hub). The two fixed UWB anchors just need power
(USB power adapter or the hub) — they don't need to stay connected to
your laptop.

## Gestures (start with this subset)

- Push, Pull — strong UWB range signature (distance to front anchor changes)
- Left, Right — strong differential signature between the two anchors
- Clockwise, Anti-clockwise — strong IMU gyro signature, weak UWB signature

Starting with 6 gestures where you can predict *which sensor should help
more* gives you a natural story for the "did fusion help all gestures,
or only some?" evaluation question.

## Workflow

1. **Flash firmware**
   - `imu_core2_firmware.ino` → Core2 board
   - `uwb_anchor_firmware.ino` → both anchor UWB modules (give each a
     unique address, see TODOs in the file)
   - `uwb_tag_firmware.ino` → the wrist-worn UWB module

   Check your UWB module's actual chip/library before flashing — the
   template assumes a DW1000-based board using the `arduino-dw1000`
   library. Swap in the matching library if your kit uses a different
   chip (ask the TA if unsure).

2. **Collect data** — one trial per run:
   ```
   python collect_data.py --imu-port /dev/tty.XXXX --uwb-port /dev/tty.YYYY \
       --person alex --gesture push --duration 2.5
   ```
   Repeat for every gesture × every group member. Aim for a balanced
   number of trials per gesture (20-30 to start).

3. **Extract features**:
   ```
   python extract_features.py
   ```
   Produces `data/features/imu_only_features.csv`,
   `uwb_only_features.csv`, and `fused_features.csv`.

4. **Train and evaluate**:
   ```
   python train_and_evaluate.py
   ```
   Compares IMU-only vs UWB-only vs Fused, KNN vs Random Forest, and
   random split vs held-out-person split. Saves a summary table and
   confusion matrix plots to `outputs/`.

## Install dependencies

```
pip install pyserial pandas numpy scikit-learn matplotlib --break-system-packages
```

## What to report (per the lab requirements)

- Confusion matrix for baseline (IMU-only) and fused model
- Which gestures are easiest/hardest, and why (tie back to whether that
  gesture has a strong signature in IMU, UWB, or both)
- Held-out-person accuracy vs random-split accuracy — does the model
  generalize to someone it hasn't seen?
- Whether fusion helped all gestures or only some (this is the natural
  finding given the gesture set above — Push/Pull/Left/Right should
  benefit from UWB, Clockwise/Anti-clockwise may not)
