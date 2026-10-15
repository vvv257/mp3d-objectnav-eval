import pytest

from objectnav_eval.adapters.observation import ObservationAdapter


def test_image_and_depth_conversion_preserve_source():
    np = pytest.importorskip("numpy")
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    depth = np.array([[[.25], [.75]]], dtype=np.float32)
    result = ObservationAdapter({
        "rgb": {"resize": [1, 2], "transpose": [2, 0, 1], "dtype": "float32"},
        "depth": {"scale": 10, "offset": 1, "clip": [0, 5]},
    }).convert({"rgb": rgb, "depth": depth})
    assert result["rgb"].shape == (3, 1, 2)
    assert result["rgb"].dtype == np.float32
    assert result["depth"].reshape(-1).tolist() == [3.5, 5]
    assert depth.reshape(-1).tolist() == [.25, .75]


def test_rename_collision_and_unknown_rule_are_rejected():
    with pytest.raises(ValueError):
        ObservationAdapter({"rgb": {"query_navmesh": True}})
    with pytest.raises(ValueError):
        ObservationAdapter({"rgb": {"rename": "depth"}}).convert({"rgb": 1, "depth": 2})
