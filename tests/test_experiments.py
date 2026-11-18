import pytest

from pts.calibrate import fit_box, DIRECTIONS
from pts.metrics import summarize


def test_calibration_rejects_disconnected_larger_levels():
    rows = []
    for axis in DIRECTIONS:
        for level, ratio in [(0.001, 1), (0.002, 1.03), (0.003, 0.9), (0.004, 1)]:
            nominal = level * 10 * (1 if axis.endswith("+") else -1)
            rows.extend({"axis": axis, "command_m_per_step": level, "nominal_m": nominal,
                         "measured_m": ratio * nominal} for _ in range(3))
    box = fit_box(rows)
    assert box["positive"] == box["negative"] == [0.002] * 3
    with pytest.raises(ValueError, match="x-"):
        fit_box([r for r in rows if r["axis"] != "x-"])


def test_metrics_include_failures_and_pool_time():
    report = summarize([
        {"method": "PTS", "trial_id": "a", "success": 1, "duration_s": 10, "control_steps": 150},
        {"method": "PTS", "trial_id": "b", "success": 0, "duration_s": 90, "control_steps": 1350},
    ])["PTS"]
    assert report["success_rate"] == 0.5
    assert report["successes_per_hour"] == 36
    assert report["successes_per_1000_steps"] == pytest.approx(2 / 3)
    assert report["mean_success_s"] == 10 and report["mean_attempt_s"] == 50
