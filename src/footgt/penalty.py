"""Penalty kicks as a zero-sum game, tested against Chiappori, Levitt & Groseclose (2002).

The kicker is the row player and maximises the scoring probability; the
goalkeeper is the column player and minimises it. Directions are always from
the kicker's point of view.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from footgt.zerosum import solve

DIRECTIONS = ("left", "middle", "right")
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "penalty_clg2002.csv"


@dataclass(frozen=True)
class PenaltyData:
    """Counts indexed [kicker direction, goalkeeper direction]."""

    shots: np.ndarray
    goals: np.ndarray
    directions: tuple[str, ...] = DIRECTIONS

    @property
    def scoring_rate(self) -> np.ndarray:
        return self.goals / self.shots

    @property
    def kicker_freq(self) -> np.ndarray:
        return self.shots.sum(axis=1) / self.shots.sum()

    @property
    def goalie_freq(self) -> np.ndarray:
        return self.shots.sum(axis=0) / self.shots.sum()

    def restrict(self, keep: tuple[str, ...]) -> PenaltyData:
        """Drop directions, e.g. ``restrict(("left", "right"))`` for the 2x2 game."""
        idx = [self.directions.index(d) for d in keep]
        sub = np.ix_(idx, idx)
        return PenaltyData(self.shots[sub], self.goals[sub], tuple(keep))


def load(path: Path | str | None = None) -> PenaltyData:
    """Read the kick counts; defaults to the CSV shipped in the repository's `data/` folder."""
    path = Path(path) if path is not None else DEFAULT_DATA
    shots = np.zeros((3, 3), dtype=int)
    goals = np.zeros((3, 3), dtype=int)
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            k, g = DIRECTIONS.index(rec["kicker"]), DIRECTIONS.index(rec["goalie"])
            shots[k, g] = int(rec["shots"])
            goals[k, g] = int(rec["goals"])
    return PenaltyData(shots, goals)


def posterior_equilibria(
    data: PenaltyData, n_draws: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate sampling noise in the payoff matrix through to the equilibrium.

    Each cell's scoring probability is only estimated (the middle-goalkeeper
    cells rest on 3 or 4 kicks). Draw every cell from its Beta posterior under
    a Jeffreys prior, Beta(goals + 1/2, misses + 1/2), solve the game for each
    draw, and return the kicker and goalkeeper strategies, one row per draw.
    """
    a = data.goals + 0.5
    b = data.shots - data.goals + 0.5
    k = len(data.directions)
    kicker = np.empty((n_draws, k))
    goalie = np.empty((n_draws, k))
    for d in range(n_draws):
        sol = solve(rng.beta(a, b))
        kicker[d], goalie[d] = sol.row, sol.col
    return kicker, goalie


def goodness_of_fit(observed_counts: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Pearson chi-square test of observed direction counts against a predicted mix.

    Treats kicks as independent draws from one common strategy. Chiappori et
    al. show that this is exactly what fails when players are heterogeneous, so
    a rejection here is evidence against the *pooled* model, not against
    minimax play by each individual.
    """
    expected = predicted * observed_counts.sum()
    res = stats.chisquare(observed_counts, expected)
    return float(res.statistic), float(res.pvalue)
