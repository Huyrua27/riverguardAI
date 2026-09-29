from riverguard.motion import MotionAnalyzer
from riverguard.prediction import HotspotPredictor
from riverguard.tracking import Tracker
from riverguard.types import Detection, Track


def test_tracker_persists_id_for_same_object():
    """A stationary object keeps its ID across frames (real IoU association)."""
    tracker = Tracker()
    dets = [Detection((0, 0, 5, 5), 0.9, "plastic_bottle")]
    t1 = tracker.update(dets)
    t2 = tracker.update([Detection((0, 0, 5, 5), 0.9, "plastic_bottle")])
    assert t1[0].track_id == t2[0].track_id
    assert t2[0].hits == 2


def test_tracker_follows_fast_small_object():
    """No box overlap between samples, but centre-distance gating + velocity
    prediction keep the same ID (small waste drifting between sampled frames)."""
    tracker = Tracker()
    ids = []
    for step in range(4):
        x = step * 15.0  # 10px box moving 15px per frame → IoU 0 between frames
        ids.append(tracker.update([Detection((x, 0, x + 10, 10), 0.9, "foam")])[0].track_id)
    assert len(set(ids)) == 1


def test_tracker_new_id_for_new_object():
    """A detection with no overlap spawns a fresh ID."""
    tracker = Tracker()
    a = tracker.update([Detection((0, 0, 5, 5), 0.9, "foam")])
    b = tracker.update([Detection((100, 100, 110, 110), 0.9, "foam")])
    assert a[0].track_id != b[0].track_id


def test_motion_needs_two_frames():
    m = MotionAnalyzer()
    tr = [Track(0, (0, 0, 4, 4), "foam", 0.8)]
    assert m.update(tr) == []          # first frame: no vector yet
    tr2 = [Track(0, (2, 0, 6, 4), "foam", 0.8)]
    vecs = m.update(tr2)
    assert vecs and vecs[0].dx == 2.0


def test_hotspot_does_not_pin_moving_objects_to_border():
    """Regression: a long alert horizon must not extrapolate pixels off-frame."""
    from riverguard.types import MotionVector

    pred = HotspotPredictor(grid_size=(10, 10), horizon_seconds=1200, fps=6.0,
                            lookahead_seconds=10)
    tracks = [Track(0, (48, 48, 52, 52), "foam", 0.9)]      # centre (50, 50)
    fc = pred.predict(tracks, [MotionVector(0, 0.5, 0.0, 0.5)], frame_shape=(100, 100))
    peak_row = max(range(10), key=lambda r: max(fc.grid[r]))
    peak_col = max(range(10), key=lambda c: fc.grid[peak_row][c])
    # 0.5 px/frame × 6 fps × 10 s = 30 px → centre moves to x=80 → column 8, not the edge.
    assert (peak_row, peak_col) == (5, 8)

    fast = pred.predict(tracks, [MotionVector(0, 50.0, 0.0, 50.0)], frame_shape=(100, 100))
    # Projection leaves the view → stays at the current cell instead of the border.
    assert fast.grid[5][5] == 1.0 and fast.grid[5][9] < 1.0


def test_hotspot_grid_normalised():
    pred = HotspotPredictor(grid_size=(4, 4), horizon_seconds=600)
    tracks = [Track(0, (10, 10, 20, 20), "plastic_bag", 0.7)]
    fc = pred.predict(tracks, [], frame_shape=(100, 100))
    peak = max(max(row) for row in fc.grid)
    assert peak <= 1.0
