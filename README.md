# distressed-mortgage-common-value-auction

Common-value auction model and Monte Carlo simulation for the secondary market
in distressed mortgage servicing rights (MSRs), calibrated to Freddie Mac
loan-level data. Econ 136 (Market Design) final project.

## Idea

The illiquidity of the distressed-MSR market is usually blamed on seller-side
adverse selection. This project reframes it as a common-value information-
aggregation problem: a pool's true default behavior is the same quantity for
every buyer, but each sees it only through a noisy private signal. The question
is whether a centralized auction aggregates those signals into a price that
tracks fundamental value better than bilateral OTC trading, and how the winner's
curse constrains that.

**Headline finding:** a centralized auction aggregates information OTC trading
cannot — but only when signals are reasonably precise. With noisy signals,
adding bidders *widens* the price-to-fundamental gap (the winner is the most
over-optimistic noise draw). The winner's curse, not adverse selection, is the
binding constraint.

## Files

- `calibrate_pools.py` — reads Freddie Mac SFLLD files, classifies loan outcomes,
  reports cross-pool default/loss dispersion (calibration: mean 0.063, sd 0.044).
- `auction_sim.py` — Monte Carlo auction vs. OTC, with double-largeness and
  robustness experiments.
- `sim_*.csv`, `simulation_output.txt` — generated results.

Pure standard-library Python. Run `calibrate_pools.py`, then `auction_sim.py`.

## Data

Freddie Mac loan-level files are not included (large + license-restricted).
Download the `sample_YYYY.zip` files for 2005–2008 from the
[Freddie Mac SFLLD portal](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
and unzip into the project folder.
