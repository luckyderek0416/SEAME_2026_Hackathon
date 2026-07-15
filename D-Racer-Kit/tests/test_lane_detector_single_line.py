import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'perception'))

from perception.lane_detector import LaneDetector


def test_process_on_empty_frame_does_not_crash():
    detector = LaneDetector()
    frame = np.zeros((160, 320, 3), dtype=np.uint8)

    result = detector.process(frame, draw_debug=False)

    lane_found, offset, num_lanes, junction, yellow_ratio, yellow_offset, curvature, yellow_crossline, fork, debug = result
    assert lane_found is False or np.isfinite(offset)
    assert np.isfinite(curvature)
    assert debug is None


def test_single_line_center_uses_steering_context():
    detector = LaneDetector()
    detector._lane_width = 80.0
    detector._last_steer = 0.4

    center = detector._single_line_center_with_context([(100.0, 0.0)], 320)

    assert abs(center - 140.0) < 1e-6


def test_single_line_center_uses_opposite_context_for_left_steer():
    detector = LaneDetector()
    detector._lane_width = 80.0
    detector._last_steer = -0.4

    center = detector._single_line_center_with_context([(100.0, 0.0)], 320)

    assert abs(center - 60.0) < 1e-6


def test_single_line_context_is_disabled_outside_roundabout_exit():
    detector = LaneDetector()
    detector._last_steer = 0.4

    assert not detector._should_use_single_line_context(0.0, False)


def test_roundabout_exit_context_is_held_after_crossline():
    detector = LaneDetector(crossline_sw_gate=1, sw_entry_frames=3)

    detector._update_roundabout_exit_context(True)
    assert detector._roundabout_exit_context is True

    detector._update_roundabout_exit_context(False)
    assert detector._roundabout_exit_context is True

    detector._update_roundabout_exit_context(False)
    assert detector._roundabout_exit_context is True

    detector._update_roundabout_exit_context(False)
    assert detector._roundabout_exit_context is False
