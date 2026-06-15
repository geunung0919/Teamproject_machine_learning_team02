from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(
    actual_delta: np.ndarray,
    pred_delta: np.ndarray,
    base_students: np.ndarray,
    actual_students: np.ndarray,
    pred_before_clip: np.ndarray,
    test_year: int
) -> dict[str, float]:
    """Calculate validation metrics for student deltas and resulting absolute levels."""
    actual_delta = actual_delta.astype(float)
    pred_delta = pred_delta.astype(float)
    base_students = base_students.astype(float)
    actual_students = actual_students.astype(float)
    pred_before_clip = pred_before_clip.astype(float)
    
    # Levels (student count) metrics
    pred_students = np.maximum(0, base_students + pred_delta)
    lvl_mae = mean_absolute_error(actual_students, pred_students)
    lvl_rmse = math.sqrt(mean_squared_error(actual_students, pred_students))
    lvl_r2 = r2_score(actual_students, pred_students) if len(np.unique(actual_students)) > 1 else np.nan
    
    lvl_denom = np.sum(np.abs(actual_students))
    lvl_wape = np.sum(np.abs(actual_students - pred_students)) / lvl_denom if lvl_denom else np.nan
    
    # Delta metrics
    dlt_mae = mean_absolute_error(actual_delta, pred_delta)
    dlt_rmse = math.sqrt(mean_squared_error(actual_delta, pred_delta))
    dlt_r2 = r2_score(actual_delta, pred_delta) if len(np.unique(actual_delta)) > 1 else np.nan
    
    dlt_denom = np.sum(np.abs(actual_delta))
    dlt_wape = np.sum(np.abs(actual_delta - pred_delta)) / dlt_denom if dlt_denom else np.nan
    
    # Clip diagnostic stats
    neg_count_before_clip = int(np.sum(base_students + pred_before_clip < 0))
    zero_count_after_clip = int(np.sum(pred_students == 0))
    
    mape_mask = actual_students > 0
    mape_safe = float(np.mean(np.abs((actual_students[mape_mask] - pred_students[mape_mask]) / actual_students[mape_mask]))) if mape_mask.any() else np.nan
    
    return {
        "fold_test_year": test_year,
        "MAE": lvl_mae,
        "RMSE": lvl_rmse,
        "R2": lvl_r2,
        "WAPE": lvl_wape,
        "Bias": float(np.mean(pred_students - actual_students)),
        "MAPE_safe": mape_safe,
        "delta_MAE": dlt_mae,
        "delta_RMSE": dlt_rmse,
        "delta_R2": dlt_r2,
        "delta_WAPE": dlt_wape,
        "prediction_negative_count_before_clip": neg_count_before_clip,
        "prediction_zero_count_after_clip": zero_count_after_clip,
    }
