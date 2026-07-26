"""
gestures.py
----------------------------------------------------
Canonical list of the 15 gesture classes for this project, shared
across every script (collect.py, extract_features.py, train.py, etc.)
so the label set never drifts between files.
"""

GESTURES = [
    "pull",
    "push",
    "clockwise",
    "counterclockwise",
    "left",
    "right",
    "bye_bye",
    "one_arm_boxing",
    "clapping",
    "two_arm_boxing",
    "t_arm",
    "raise_arms",
    "soli",
    "open_close_fist",
    "palm_up_down",
    "rest",
]

# Which sensor(s) are expected to be most discriminative for each
# gesture, based on the sensing modalities in this project (imu, uwb,
# mmwave, wifi, rfid). This is a HYPOTHESIS to test in evaluate.py, not
# a restriction on data collection -- collect.py records from every
# active sensor for every gesture regardless of this table, so a single
# fused model can be trained across the whole gesture set.
EXPECTED_STRONGEST_SENSOR = {
    "pull": ["imu", "uwb"],
    "push": ["imu", "uwb"],
    "clockwise": ["imu", "uwb"],
    "counterclockwise": ["imu", "uwb"],
    "left": ["imu", "uwb"],
    "right": ["imu", "uwb"],
    "bye_bye": ["imu", "mmwave"],
    "one_arm_boxing": ["imu", "mmwave"],
    "clapping": ["uwb", "mmwave"],
    "two_arm_boxing": ["imu", "mmwave"],
    "t_arm": ["uwb", "mmwave"],
    "raise_arms": ["imu", "mmwave"],
    "soli": ["mmwave"],  # originally hypothesized rfid+mmwave, but RFID
                          # wasn't reliable enough to use for this gesture
                          # in practice -- relying on mmwave alone instead
    "open_close_fist": ["rfid"],
    "palm_up_down": ["mmwave", "imu"],
}

# Discrete (one-shot) vs periodic (repeated/continuous) gesture typing,
# per the final project design hints. This matters for how a gesture
# should be performed during collection: discrete gestures should be
# done ONCE near the center of the recording window, periodic gestures
# should be repeated naturally throughout the window. It also matters
# for evaluation -- if you ever collect long continuous recordings of
# a periodic gesture, split train/test by full recording or by
# collector, never by overlapping windows from the same recording.
GESTURE_TYPE = {
    "pull": "discrete",
    "push": "discrete",
    "clockwise": "discrete",
    "counterclockwise": "discrete",
    "left": "discrete",
    "right": "discrete",
    "bye_bye": "periodic",
    "one_arm_boxing": "periodic",
    "clapping": "periodic",
    "two_arm_boxing": "periodic",
    "t_arm": "discrete",  # held pose, not repeated motion
    "raise_arms": "discrete",
    "soli": "periodic",
    "open_close_fist": "periodic",
    "palm_up_down": "periodic",
}
