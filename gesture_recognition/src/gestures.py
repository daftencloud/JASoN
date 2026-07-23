"""
gestures.py
----------------------------------------------------
Canonical list of the 15 gesture classes for this project, shared
across every script (collect.py, extract_features.py, train.py, etc.)
so the label set never drifts between files.
"""

GESTURES = [
    "rest",
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
