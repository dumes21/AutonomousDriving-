# These results are PROVISIONAL and from a mapping that has since been removed

Everything in this folder (and its sub-folders) was produced by the first version of `car/camera_test_dqn.py`, which fed the DQN
`[offset_norm, heading_slope / pi, previous_steering]` taken straight from `car/lane_detection.py`. Those are image-space proxies,
not the simulator's offset in metres or heading in radians, so the predicted actions here say nothing about driving behaviour.

`car/camera_test_dqn.py` now requires a complete `--calibration CAL.json` and uses `perception/state_estimator.py` (the same
estimator as `car/drive_dqn.py`). Re-run it once a real calibration exists (command in `docs/CALIBRATION_DAY.md`, section 7) into a NEW folder.
`selected_categories.csv` (the 46-frame sample) can be reused as is.
