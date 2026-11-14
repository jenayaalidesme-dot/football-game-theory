import numpy as np
import pytest

from footgt import marking, solve


def test_default_matrix_matches_the_hand_derivation():
    s = 2 * np.sqrt(2)
    expected = np.array([[2, 5, 5], [5, s, 5], [5, 5, s]])
    assert marking.payoff_matrix() == pytest.approx(expected)


def test_equilibrium_matches_the_original_result():
    # Reported as (0.26, 0.37, 0.37) in the 2024 presentation.
    sol = solve(marking.payoff_matrix())
    assert sol.row == pytest.approx([0.266, 0.367, 0.367], abs=1e-3)
    assert sol.col == pytest.approx(sol.row)  # the game is symmetric
    assert sol.row[1] == pytest.approx(sol.row[2])  # left and right are interchangeable


def test_no_interception_when_players_cannot_meet():
    assert (marking.payoff_matrix(distance=10.0, step=2.0, escape_gain=5.0) == 5.0).all()


def test_rejects_step_longer_than_gap():
    with pytest.raises(ValueError):
        marking.payoff_matrix(distance=1.0, step=2.0)


def test_simulated_gain_converges_to_game_value():
    rng = np.random.default_rng(42)
    payoff = marking.payoff_matrix()
    sol = solve(payoff)
    gains = marking.simulate(payoff, sol.row, sol.col, 200_000, rng)
    stderr = gains.std(ddof=1) / np.sqrt(gains.size)
    assert abs(gains.mean() - sol.value) < 4 * stderr


def test_equilibrium_attacker_cannot_be_exploited_by_any_pure_defence():
    rng = np.random.default_rng(3)
    payoff = marking.payoff_matrix()
    sol = solve(payoff)
    for j in range(3):
        pure = np.eye(3)[j]
        gains = marking.simulate(payoff, sol.row, pure, 200_000, rng)
        stderr = gains.std(ddof=1) / np.sqrt(gains.size)
        assert abs(gains.mean() - sol.value) < 4 * stderr
