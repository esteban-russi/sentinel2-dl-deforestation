import numpy as np

from change_detection import (
    reclassify_to_forest_binary,
    compute_change_map,
    quantify_change,
    format_change_report,
)


def test_reclassify_to_forest_binary():
    # 4 classes: 0 (Other), 1 (Trees/Forest), 2 (Grass), 3 (Shrub)
    preds = np.array([
        [0, 1, 2],
        [3, 1, 0]
    ])
    binary = reclassify_to_forest_binary(preds, forest_class_index=1)
    expected = np.array([
        [0, 1, 0],
        [0, 1, 0]
    ])
    np.testing.assert_array_equal(binary, expected)


def test_compute_change_map():
    forest_t1 = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0]
    ])
    forest_t2 = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 1]
    ])
    # T2 - T1:
    # [0-1=-1 (Deforest), 1-0=1 (Reforest), 1-1=0 (No change), 0-0=0 (No change)]
    # [0-1=-1 (Deforest), 1-1=0 (No change), 0-0=0 (No change), 1-0=1 (Reforest)]
    expected_change = np.array([
        [-1,  1,  0,  0],
        [-1,  0,  0,  1]
    ])
    change_map = compute_change_map(forest_t1, forest_t2)
    np.testing.assert_array_equal(change_map, expected_change)


def test_quantify_change():
    change_map = np.array([
        [-1, -1, 1, 0],
        [ 0,  0, 0, 0]
    ])  # 2 deforested, 1 reforested, total 8 pixels
    # With resolution 10m:
    # 100 m^2 per pixel
    # 2 pixels = 200 m^2 = 0.0002 km^2
    # 1 pixel = 100 m^2 = 0.0001 km^2
    # 8 pixels = 800 m^2 = 0.0008 km^2
    metrics = quantify_change(change_map, pixel_resolution_m=10)
    assert metrics['pixels_deforested'] == 2
    assert metrics['pixels_reforested'] == 1
    assert metrics['total_pixels'] == 8
    assert np.isclose(metrics['area_deforested_sq_km'], 0.0002)
    assert np.isclose(metrics['area_reforested_sq_km'], 0.0001)
    assert np.isclose(metrics['total_area_sq_km'], 0.0008)
    assert np.isclose(metrics['net_change_sq_km'], -0.0001)


def test_format_change_report():
    metrics = {
        'pixels_deforested': 1000,
        'pixels_reforested': 250,
        'total_pixels': 10000,
        'area_deforested_sq_km': 10.0,
        'area_reforested_sq_km': 2.5,
        'total_area_sq_km': 100.0,
        'net_change_sq_km': -7.5,
    }
    report = format_change_report(metrics)
    assert "Change Detection Analysis Report" in report
    assert "Total Analyzed Area: 100.00 km²" in report
    assert "Pixels Changed: 1,000" in report
    assert "Area Changed:   10.00 km²" in report
    assert "Pixels Changed: 250" in report
    assert "Area Changed:   2.50 km²" in report
