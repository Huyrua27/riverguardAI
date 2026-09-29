from riverguard.utils.metrics import iou, mae, rmse


def test_iou_identical_boxes():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes():
    assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_iou_half_overlap():
    assert abs(iou((0, 0, 2, 2), (1, 0, 3, 2)) - (2 / 6)) < 1e-9


def test_mae_rmse_zero_error():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_known():
    assert mae([1.0, 2.0], [2.0, 4.0]) == 1.5
