import pytest
import pandas as pd
import numpy as np
# Import your functions from your main script
from utils_vis_PID import calculate_mape, calculate_smape, calculate_mae, calculate_rmse, calculate_nmape_shifted

@pytest.fixture
def test_speed_mape_with_traffic_jam():
    # Ground truth (sim) is 0 (standstill), PID predicts 20 (free flow)
    sim = pd.DataFrame([[0.0]])
    pid = pd.DataFrame([[20.0]])
    
    # With eps=1.0: |0 - 20| / (0 + 1) = 20 -> 2000%
    error = calculate_mape(sim, pid, eps=1.0)
    
    assert error == 2000.0
    assert not np.isinf(error)
    assert not np.isnan(error)

def test_speed_mape_perfect_jam():
    # Both are 0 (both agree it's a standstill)
    sim = pd.DataFrame([[0.0]])
    pid = pd.DataFrame([[0.0]])
    
    # |0 - 0| / (0 + 1) = 0
    error = calculate_mape(sim, pid, eps=1.0)
    assert error == 0.0


# --- Flow MAPE Tests (Using eps=100) ---

def test_flow_mape_low_volume_noise():
    """Tests that eps=100 suppresses noise in low flow conditions."""
    # Sim is 10 vph, PID predicts 60 vph (a 500% raw error)
    sim = pd.DataFrame([[10.0]])
    pid = pd.DataFrame([[60.0]])
    
    # Calculation: |10 - 60| / (10 + 100) = 50 / 110 approx 0.4545
    # Result should be approx 45.45%
    error = calculate_mape(sim, pid, eps=100.0)
    
    assert error == pytest.approx(45.4545, rel=1e-3)

def test_flow_mape_high_volume():
    """Tests that eps=100 has minimal impact on high flow results."""
    # Sim is 2000 vph, PID predicts 1800 vph
    sim = pd.DataFrame([[2000.0]])
    pid = pd.DataFrame([[1800.0]])
    
    # Calculation: |2000 - 1800| / (2000 + 100) = 200 / 2100 approx 0.0952
    # Result should be approx 9.52%
    error = calculate_mape(sim, pid, eps=100.0)
    
    assert error == pytest.approx(9.5238, rel=1e-3)


# --- Symmetric MAPE Tests (No eps needed) ---

def test_smape_standstill_failure():
    sim = pd.DataFrame([[0.0]])
    pid = pd.DataFrame([[40.0]])
    error = calculate_smape(sim, pid)
    # Use approx because 1e-5 makes it 199.999...
    assert error == pytest.approx(200.0, abs=1e-4)

def test_smape_symmetry():
    sim_a, pid_a = pd.DataFrame([[10.0]]), pd.DataFrame([[30.0]])
    sim_b, pid_b = pd.DataFrame([[30.0]]), pd.DataFrame([[10.0]])
    error_a = calculate_smape(sim_a, pid_a)
    error_b = calculate_smape(sim_b, pid_b)
    assert error_a == error_b
    assert error_a == pytest.approx(100.0, abs=1e-4)

# --- Updated Flow MAPE Tests ---
# Check your calculate_mape to ensure it is (s-p)/s and not (s-p)/p

def test_flow_mape_low_volume_noise():
    sim = pd.DataFrame([[10.0]])
    pid = pd.DataFrame([[60.0]])
    # |10-60| / (10+100) = 50/110 = 0.4545
    error = calculate_mape(sim, pid, eps=100.0)
    assert error == pytest.approx(45.4545, rel=1e-3)

def test_mape_nan_exclusion():
    # sim[10, NaN], pid[20, 10]
    # Valid pair is (10, 20). 
    # Calculation: |10-20| / (10+1) = 10/11 = 90.909%
    sim = pd.DataFrame([[10.0, np.nan]])
    pid = pd.DataFrame([[20.0, 10.0]])
    error = calculate_mape(sim, pid, eps=1.0)
    assert error == pytest.approx(90.909, rel=1e-3)

def test_mae_basic():
    """Checks simple mean absolute difference."""
    sim = pd.DataFrame([[10.0, 20.0]])
    pid = pd.DataFrame([[15.0, 25.0]])
    # (|10-15| + |20-25|) / 2 = 5.0
    assert calculate_mae(sim, pid) == 5.0

def test_mae_with_nans():
    """Ensures MAE ignores misaligned NaNs."""
    sim = pd.DataFrame([[10.0, np.nan]])
    pid = pd.DataFrame([[15.0, 100.0]])
    # Only compares (10, 15) -> 5.0
    assert calculate_mae(sim, pid) == 5.0

# --- RMSE Tests ---
def test_rmse_outlier_sensitivity():
    """RMSE should be higher than MAE when one large error exists."""
    sim = pd.DataFrame([[10.0, 10.0]])
    pid = pd.DataFrame([[10.0, 20.0]]) 
    # Error vector: [0, 10]. Mean Sq: (0^2 + 10^2)/2 = 50. Sqrt(50) approx 7.07
    mae = calculate_mae(sim, pid)  # 5.0
    rmse = calculate_rmse(sim, pid) # 7.07
    assert rmse > mae
    assert rmse == pytest.approx(7.071, rel=1e-3)

# --- NMAPE (Shifted/Normalized) Tests ---
def test_nmape_zero_handling():
    """Tests the logic: |0-20| / (0+1) = 20 (2000%) vs |0-20| / (0+0.01) = 2000 (200,000%)"""
    sim = pd.DataFrame([[0.0]])
    pid = pd.DataFrame([[20.0]])
    
    # Coworker version (Shifted by 1.0)
    nmape_val = calculate_nmape_shifted(sim, pid, shift=1.0)
    # Standard MAPE with your chosen eps=0.01
    mape_val = calculate_mape(sim, pid, eps=0.01)
    
    assert nmape_val == 2000.0   # Much more 'readable'
    assert mape_val == 200000.0  # Mathematically raw, but huge

# --- Edge Case: Empty Intersection ---
def test_all_metrics_empty():
    sim = pd.DataFrame([[np.nan]])
    pid = pd.DataFrame([[10.0]])
    assert np.isnan(calculate_mae(sim, pid))
    assert np.isnan(calculate_rmse(sim, pid))
    assert np.isnan(calculate_nmape_shifted(sim, pid))
    assert np.isnan(calculate_mape(sim, pid, eps=1.0))
    assert np.isnan(calculate_smape(sim, pid))