"""One-on-one marking duel: an attacker runs at a defender standing `distance` metres away.

Both choose simultaneously between going straight ("A", for *avant*) or
cutting 45 degrees to the attacker's left ("G") or right ("D"). A step covers
`step` metres of ground along the attacker–defender axis, so a diagonal
step is `step * sqrt(2)` long. If both end up on the same spot the defender
wins the ball; otherwise the attacker escapes.

The attacker's payoff is the distance covered while still in possession: the
length of the step if intercepted, `escape_gain` if not. The attacker is the
row player (maximiser).

All directions are expressed in the attacker's frame, so "the defender plays G"
means the defender covers the attacker's left.
"""

from __future__ import annotations

import numpy as np

MOVES = ("A", "G", "D")
_LATERAL = {"A": 0.0, "G": 1.0, "D": -1.0}


def endpoint(move: str, step: float) -> tuple[float, float]:
    """(forward, lateral) displacement of one move."""
    return step, _LATERAL[move] * step


def payoff_matrix(distance: float = 4.0, step: float = 2.0, escape_gain: float = 5.0) -> np.ndarray:
    """Attacker's payoff, indexed [attacker move, defender move]."""
    if not 0 < step <= distance:
        raise ValueError("need 0 < step <= distance")
    payoff = np.empty((3, 3))
    for i, a in enumerate(MOVES):
        ax, ay = endpoint(a, step)
        for j, d in enumerate(MOVES):
            dx, dy = endpoint(d, step)
            # The defender starts `distance` ahead and moves towards the attacker.
            met = np.isclose(ax, distance - dx) and np.isclose(ay, dy)
            payoff[i, j] = float(np.hypot(ax, ay)) if met else escape_gain
    return payoff


def simulate(
    payoff: np.ndarray,
    attacker: np.ndarray,
    defender: np.ndarray,
    n_duels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Play `n_duels` independent duels and return the attacker's gain in each."""
    i = rng.choice(len(attacker), size=n_duels, p=attacker)
    j = rng.choice(len(defender), size=n_duels, p=defender)
    return payoff[i, j]
