import numpy as np
import pytest

from footgt import exploitability, fictitious_play, solve


def test_matching_pennies():
    sol = solve(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert sol.row == pytest.approx([0.5, 0.5])
    assert sol.col == pytest.approx([0.5, 0.5])
    assert sol.value == pytest.approx(0.0, abs=1e-9)


def test_rock_paper_scissors():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    sol = solve(rps)
    assert sol.row == pytest.approx([1 / 3] * 3)
    assert sol.col == pytest.approx([1 / 3] * 3)


def test_saddle_point_gives_pure_strategies():
    # Entry (1, 0) is the minimum of its row and the maximum of its column.
    game = np.array([[1.0, 5.0], [3.0, 4.0]])
    sol = solve(game)
    assert sol.row == pytest.approx([0.0, 1.0], abs=1e-9)
    assert sol.col == pytest.approx([1.0, 0.0], abs=1e-9)
    assert sol.value == pytest.approx(3.0)


def test_two_by_two_closed_form():
    a, b, c, d = 0.632, 0.941, 0.895, 0.440
    sol = solve(np.array([[a, b], [c, d]]))
    denom = a - b - c + d
    assert sol.row[0] == pytest.approx((d - c) / denom)
    assert sol.col[0] == pytest.approx((d - b) / denom)
    assert sol.value == pytest.approx((a * d - b * c) / denom)


@pytest.mark.parametrize("seed", range(20))
def test_random_games_have_zero_exploitability(seed):
    rng = np.random.default_rng(seed)
    game = rng.normal(size=(rng.integers(2, 7), rng.integers(2, 7)))
    sol = solve(game)
    assert sol.row.sum() == pytest.approx(1.0)
    assert sol.col.sum() == pytest.approx(1.0)
    assert (sol.row >= 0).all() and (sol.col >= 0).all()
    assert exploitability(game, sol.row, sol.col) == pytest.approx(0.0, abs=1e-8)
    assert sol.row @ game @ sol.col == pytest.approx(sol.value, abs=1e-8)


def test_exploitability_is_positive_off_equilibrium():
    pennies = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert exploitability(pennies, np.array([0.9, 0.1]), np.array([0.5, 0.5])) > 0.5


def test_fictitious_play_agrees_with_linear_programme():
    rng = np.random.default_rng(0)
    game = np.array([[2.0, 5.0, 5.0], [5.0, 2.83, 5.0], [5.0, 5.0, 2.83]])
    row, col, gap = fictitious_play(game, 20_000, rng)
    sol = solve(game)
    assert gap[-1] < 0.02
    assert gap[-1] < gap[100]
    assert row == pytest.approx(sol.row, abs=0.02)
    assert col == pytest.approx(sol.col, abs=0.02)


@pytest.mark.parametrize("bad", [np.zeros(3), np.empty((0, 2)), np.array([[np.nan, 1.0]])])
def test_rejects_malformed_payoffs(bad):
    with pytest.raises(ValueError):
        solve(bad)
