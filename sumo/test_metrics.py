import pytest
import pandas as pd
import numpy as np
# Import your functions from your main script
from utils_vis_PID import calculate_mape, calculate_smape 

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

def test_metrics_empty_intersection():
    """Tests behavior when no overlapping valid data exists."""
    # No indices where both have non-NaN values
    sim = pd.DataFrame([[10.0, np.nan]])
    pid = pd.DataFrame([[np.nan, 10.0]])
    
    mape_err = calculate_mape(sim, pid)
    smape_err = calculate_smape(sim, pid)
    
    # np.mean([]) returns np.nan, which is a correct way to signal 'no data'
    assert np.isnan(mape_err)
    assert np.isnan(smape_err)