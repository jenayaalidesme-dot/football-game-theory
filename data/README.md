# Data

`penalty_clg2002.csv` — 459 penalty kicks from the French (1997–1999) and Italian
(1997–2000) first leagues, aggregated by goalkeeper and kicker direction.

Source: P.-A. Chiappori, S. Levitt and T. Groseclose, "Testing Mixed-Strategy
Equilibria When Players Are Heterogeneous: The Case of Penalty Kicks in Soccer",
*American Economic Review* 92(4), 2002, Tables 3 and 4.

- `shots` is Table 3 (number of kicks per cell).
- `goals` is reconstructed as `round(shots × scoring rate)` from Table 4, which
  reports percentages to one decimal. `tests/test_penalty.py` checks that every
  reconstructed count reproduces the published percentage.
- Directions follow the paper's convention: "left" and "right" are both from the
  kicker's point of view, so `left,left` means the goalkeeper guessed correctly.
