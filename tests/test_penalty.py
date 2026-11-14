import numpy as np
import pytest

from footgt import penalty, solve

# Table 4 of Chiappori et al. (2002), percentage of kicks scored, [goalie, kicker].
PUBLISHED_RATES = np.array([[63.2, 81.2, 89.5], [100.0, 0.0, 100.0], [94.1, 89.3, 44.0]])


@pytest.fixture(scope="module")
def data():
    return penalty.load()


def test_counts_match_the_paper(data):
    assert data.shots.sum() == 459
    assert data.shots.sum(axis=1).tolist() == [206, 79, 174]  # kicker totals
    assert data.shots.sum(axis=0).tolist() == [260, 11, 188]  # goalkeeper totals


def test_reconstructed_goals_reproduce_published_percentages(data):
    assert np.round(100 * data.scoring_rate.T, 1) == pytest.approx(PUBLISHED_RATES)
    assert round(100 * data.goals.sum() / data.shots.sum(), 1) == 74.9


def test_equilibrium_equalises_scoring_rate_across_directions_used(data):
    rate = data.scoring_rate
    sol = solve(rate)
    per_direction = rate @ sol.col  # kicker's scoring probability for each direction
    assert per_direction[sol.row > 1e-9] == pytest.approx(sol.value)


def test_restrict_to_two_directions(data):
    two = data.restrict(("left", "right"))
    assert two.shots.shape == (2, 2)
    assert two.shots.sum() == 117 + 95 + 85 + 75
    sol = solve(two.scoring_rate)
    assert sol.row[0] == pytest.approx(0.596, abs=2e-3)


def test_posterior_draws_are_valid_strategies(data):
    kicker, goalie = penalty.posterior_equilibria(data, 200, np.random.default_rng(1))
    assert kicker.shape == goalie.shape == (200, 3)
    assert kicker.sum(axis=1) == pytest.approx(np.ones(200))
    assert goalie.sum(axis=1) == pytest.approx(np.ones(200))
    assert (kicker >= 0).all() and (goalie >= 0).all()


def test_posterior_is_reproducible_from_the_seed(data):
    a = penalty.posterior_equilibria(data, 20, np.random.default_rng(7))
    b = penalty.posterior_equilibria(data, 20, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_goodness_of_fit_accepts_the_true_mix():
    stat, p = penalty.goodness_of_fit(np.array([300, 200, 500]), np.array([0.3, 0.2, 0.5]))
    assert stat == pytest.approx(0.0)
    assert p == pytest.approx(1.0)
