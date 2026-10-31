import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from prism.control.contract import ControlContract, MotionStats, project_map


def apply(contract, index, **kwargs):
    return contract.apply(np.full((3, 4), -6.), np.full((8, 8), 8.),
        np.zeros((36, 64)), np.zeros((3, 4), bool), frame_index=index, base_qp=32, **kwargs)


@pytest.mark.smoke
@pytest.mark.parametrize("trigger", [{"scene_cut": True}, {"velocity_1080p": 15.01}])
def test_st10_immediate_zero_held_three_frames_and_history_flush(trigger):
    contract = ControlContract()
    assert np.any(apply(contract, 0).delta_qp)
    for index in range(1, 4):
        result = apply(contract, index, **(trigger if index == 1 else {}))
        assert not np.any(result.delta_qp) and not np.any(result.residual)
        assert contract.previous_map is None
    assert np.any(apply(contract, 4).delta_qp)


@pytest.mark.smoke
def test_st01_frame_bounds_and_cqp_budget_off():
    contract = ControlContract(zero_sum=True)
    result = apply(contract, 0, measured_delta_bits=lambda _: (_ for _ in ()).throw(AssertionError()))
    assert not result.log["budget_on"] and np.max(result.residual) == 3
    assert np.max(apply(contract, 1, frame_type="non_reference_B").residual) == 8


def test_st02_reject_future_motion_and_preserve_causal_lag():
    contract = ControlContract()
    apply(contract, 0)
    bad = MotionStats(1, np.zeros((3, 4, 2)), 1080, 1920)
    with pytest.raises(ValueError, match="previous"):
        apply(contract, 1, previous_motion=bad)


def test_projection_rechecks_rounded_budget_never_breaks_floor():
    raw = np.array([[-6., 4.], [-6., 4.]])
    bg = np.array([[False, True], [False, True]])
    # QP changes map to a nonlinear measured-rate model; sum(QP)==0 is insufficient.
    budget = lambda values: float(np.sum(2 ** (-values / 6) - 1))
    result, bypass = project_map(raw, bg, 32, budget)
    assert not bypass and budget(result) <= 0
    assert np.all(result[bg] <= 4) and np.all(np.abs(result) <= 6)
    assert np.array_equal(result, np.rint(result))


def test_infeasible_budget_breaker_zeros_both_heads():
    contract = ControlContract(zero_sum=True)
    for index in range(3):
        result = apply(contract, index, rc_mode="ABR", measured_delta_bits=lambda _: 1.)
        assert result.log["bypass"] and not np.any(result.delta_qp) and not np.any(result.residual)


@pytest.mark.parametrize("base", [0, 1, 50, 51])
def test_legal_qp_at_edges(base):
    raw = np.array([[-6., 6.]])
    result, _ = project_map(raw, np.ones_like(raw, bool), base, None)
    assert np.all((result + base >= 0) & (result + base <= 51))
