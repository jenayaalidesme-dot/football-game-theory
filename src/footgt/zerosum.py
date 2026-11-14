"""Two-player zero-sum matrix games.

Convention used everywhere in this package: ``payoff[i, j]`` is what the *row*
player receives (and the column player loses) when row plays ``i`` and column
plays ``j``. Row maximises, column minimises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Solution:
    """Minimax solution of a zero-sum game."""

    row: np.ndarray  # row player's equilibrium mixed strategy
    col: np.ndarray  # column player's equilibrium mixed strategy
    value: float  # expected payoff to the row player at equilibrium


def _maximin(payoff: np.ndarray) -> tuple[np.ndarray, float]:
    """Row player's maximin strategy: max_x min_j (x @ payoff)[j].

    Linear programme in (x, v): maximise v subject to x @ payoff[:, j] >= v for
    every column j, with x on the probability simplex.
    """
    m, n = payoff.shape
    cost = np.r_[np.zeros(m), -1.0]
    a_ub = np.c_[-payoff.T, np.ones(n)]
    a_eq = np.r_[np.ones(m), 0.0][None, :]
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(cost, A_ub=a_ub, b_ub=np.zeros(n), A_eq=a_eq, b_eq=[1.0], bounds=bounds)
    if not res.success:  # pragma: no cover - a finite matrix game always has a solution
        raise RuntimeError(f"linear programme failed: {res.message}")
    x = np.clip(res.x[:m], 0.0, None)
    return x / x.sum(), float(res.x[m])


def solve(payoff: np.ndarray) -> Solution:
    """Solve a zero-sum matrix game by linear programming (von Neumann's minimax theorem)."""
    payoff = np.asarray(payoff, dtype=float)
    if payoff.ndim != 2 or payoff.size == 0:
        raise ValueError("payoff must be a non-empty 2-D array")
    if not np.all(np.isfinite(payoff)):
        raise ValueError("payoff must be finite")
    row, value = _maximin(payoff)
    col, _ = _maximin(-payoff.T)
    return Solution(row=row, col=col, value=value)


def exploitability(payoff: np.ndarray, row: np.ndarray, col: np.ndarray) -> float:
    """How much the two players could gain in total by best-responding.

    Zero exactly at a Nash equilibrium, positive otherwise. This is the check
    that does not depend on how the equilibrium was computed.
    """
    payoff = np.asarray(payoff, dtype=float)
    best_row = np.max(payoff @ col)  # best the row player can get against `col`
    best_col = np.min(row @ payoff)  # best (lowest) the column player can hold `row` to
    return float(best_row - best_col)


def fictitious_play(
    payoff: np.ndarray, n_iter: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brown's fictitious play: each round, best-respond to the opponent's past frequencies.

    Returns the empirical row and column frequencies and the exploitability
    after each round. Robinson (1951) proved the frequencies converge to an
    equilibrium in zero-sum games, so this is an independent check on `solve`.
    """
    payoff = np.asarray(payoff, dtype=float)
    m, n = payoff.shape
    row_counts = np.zeros(m)
    col_counts = np.zeros(n)
    row_counts[rng.integers(m)] += 1
    col_counts[rng.integers(n)] += 1
    gap = np.empty(n_iter)
    for t in range(n_iter):
        i = int(np.argmax(payoff @ col_counts))
        j = int(np.argmin(row_counts @ payoff))
        row_counts[i] += 1
        col_counts[j] += 1
        gap[t] = exploitability(
            payoff, row_counts / row_counts.sum(), col_counts / col_counts.sum()
        )
    return row_counts / row_counts.sum(), col_counts / col_counts.sum(), gap
