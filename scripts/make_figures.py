"""Regenerate every figure and number quoted in the README.

python scripts/make_figures.py [--out figures] [--seed 2026]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from footgt import fictitious_play, marking, penalty, solve

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)


def penalty_figure(out: Path, rng: np.random.Generator, n_draws: int) -> None:
    data = penalty.load()
    sol = solve(data.scoring_rate)
    kicker, goalie = penalty.posterior_equilibria(data, n_draws, rng)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    panels = [
        ("Kicker", sol.row, kicker, data.kicker_freq, data.shots.sum(axis=1)),
        ("Goalkeeper", sol.col, goalie, data.goalie_freq, data.shots.sum(axis=0)),
    ]
    x = np.arange(3)
    print("Penalty kicks, 3x3 game (Chiappori et al. 2002 data)")
    print(
        f"  value of the game: {sol.value:.3f}  (observed scoring rate "
        f"{data.goals.sum() / data.shots.sum():.3f})"
    )
    for ax, (who, point, draws, observed, counts) in zip(axes, panels, strict=True):
        lo, hi = np.percentile(draws, [5, 95], axis=0)
        ax.bar(x - 0.2, point, 0.36, color=BLUE, label="Minimax prediction")
        ax.bar(x + 0.2, observed, 0.36, color=ORANGE, label="Observed (459 kicks)")
        ax.errorbar(
            x - 0.2,
            point,
            yerr=[np.clip(point - lo, 0, None), np.clip(hi - point, 0, None)],
            fmt="none",
            ecolor=INK,
            elinewidth=1,
            capsize=3,
        )
        for xi, p, o in zip(x, point, observed, strict=True):
            ax.text(xi + 0.2, o + 0.015, f"{o:.2f}", ha="center", color=INK, fontsize=9)
            ax.text(
                xi - 0.2, max(p, hi[xi]) + 0.015, f"{p:.2f}", ha="center", color=INK, fontsize=9
            )
        ax.set_xticks(x, [d.capitalize() for d in data.directions])
        ax.set_title(who)
        ax.set_ylim(0, 0.9)
        stat, pval = penalty.goodness_of_fit(counts, point)
        print(
            f"  {who:<10} predicted {np.round(point, 3)}  90% interval low {np.round(lo, 3)} "
            f"high {np.round(hi, 3)}"
        )
        print(f"  {'':<10} observed  {np.round(observed, 3)}  chi2 = {stat:.1f}, p = {pval:.2g}")
    axes[0].set_ylabel("Share of kicks")
    axes[1].legend(loc="upper right")
    fig.suptitle(
        "Penalty kicks: minimax prediction against observed play",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Whiskers: 90% posterior interval from sampling noise in the scoring rates. "
        "Directions from the kicker's side.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "penalty_prediction_vs_observed.png", dpi=160)
    plt.close(fig)

    two = data.restrict(("left", "right"))
    s2 = solve(two.scoring_rate)
    print(f"  2x2 game (no middle): kicker left {s2.row[0]:.3f}, goalkeeper right {s2.col[1]:.3f}")


def marking_figure(out: Path, rng: np.random.Generator) -> None:
    payoff = marking.payoff_matrix()
    sol = solve(payoff)
    print("Marking duel (distance 4 m, step 2 m, escape gain 5 m)")
    print(
        f"  attacker {np.round(sol.row, 3)}  defender {np.round(sol.col, 3)}"
        f"  value {sol.value:.3f} m"
    )
    gains = marking.simulate(payoff, sol.row, sol.col, 100_000, rng)
    print(
        f"  Monte Carlo, 100 000 duels: mean gain {gains.mean():.3f} "
        f"± {1.96 * gains.std(ddof=1) / np.sqrt(gains.size):.3f} m"
    )

    escape = np.linspace(2.9, 12, 120)
    straight = np.array([solve(marking.payoff_matrix(escape_gain=g)).row[0] for g in escape])
    _, _, gap = fictitious_play(payoff, 5000, rng)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(escape, straight, color=BLUE, lw=2)
    ax1.plot([5], [sol.row[0]], "o", color=BLUE, ms=8, mec="#fcfcfb", mew=2)
    ax1.annotate(
        f"model setting: {sol.row[0]:.2f}",
        (5, sol.row[0]),
        xytext=(5.6, sol.row[0] - 0.05),
        color=INK,
        fontsize=9,
    )
    ax1.axhline(1 / 3, color=MUTED, lw=1, ls=":")
    ax1.text(12, 1 / 3 + 0.008, "uniform play (1/3)", ha="right", color=MUTED, fontsize=8)
    ax1.set_xlabel("Gain when the attacker escapes (m)")
    ax1.set_ylabel("P(attacker goes straight)")
    ax1.set_title("Equilibrium depends on the escape gain")
    ax1.set_ylim(0, 0.4)

    ax2.loglog(np.arange(1, gap.size + 1), gap, color=BLUE, lw=2)
    ax2.set_xlabel("Rounds of fictitious play")
    ax2.set_ylabel("Exploitability (m)")
    ax2.set_title("Learning dynamics reach the same equilibrium")
    fig.tight_layout()
    fig.savefig(out / "marking_equilibrium.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    penalty_figure(args.out, rng, args.draws)
    marking_figure(args.out, rng)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
