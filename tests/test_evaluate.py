import numpy as np
from stresstest_fin.evaluate import expected_calibration_error, predictive_metrics

def test_ece_is_bounded():
    assert 0 <= expected_calibration_error(np.array([0,1,0,1]), np.array([.1,.9,.2,.8])) <= 1

def test_metrics_shape():
    report = predictive_metrics(np.array([0,1,0,1]), np.array([.1,.9,.3,.8]))
    assert "roc_auc" in report and report["roc_auc"] >= 0
