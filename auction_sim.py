#!/usr/bin/env python3
"""
auction_sim.py
--------------
Monte Carlo simulation for the Econ 136 MSR project.

Question: can a centralized common-value auction aggregate dispersed buyer
signals about a distressed pool's true value into a clearing price that tracks
fundamental value, and how does the winner's curse constrain that as a function
of the number of bidders n and the signal precision?

Setup (mixed common-plus-private value, per instructor guidance):
  - A pool's COMMON value driver is its true default rate D, calibrated to the
    Freddie Mac SFLLD pool data: mean 0.063, sd 0.044, bounded in (0,1).
    D is drawn from a Beta distribution matched to that mean and sd.
  - The pool's economic value to the common component is V = price of a clean
    pool minus expected loss = 1 - LGD * D  (per dollar of UPB), with LGD=0.41
    (the realized loss-given-default implied by the Freddie data: loss rate /
    default rate = 0.0259 / 0.063).  Higher default -> lower value.
  - Each buyer i sees a noisy signal s_i = D + eps_i, eps_i ~ N(0, sigma^2),
    and a small private term c_i ~ N(0, tau^2) (servicing-cost heterogeneity).
    Buyer valuation: v_i = (1 - LGD * D) + beta * c_i, with beta small.

Mechanisms compared on identical draws:
  - UNIFORM-PRICE AUCTION with winner's-curse-aware bid shading. Each buyer
    conditions on the event of being pivotal (signal is the max), forming
    E[v_i | s_i, s_i = max], which shades the naive estimate downward.
  - OTC BENCHMARK: a single randomly chosen buyer negotiates bilaterally; the
    transaction "price" reflects only that one buyer's signal (no aggregation),
    with split-the-surplus pricing against the seller's reservation.

Metrics (averaged over many repetitions, per (n, sigma) cell):
  - price_to_fundamental_gap: |clearing price - true V| (lower = better
    aggregation). Reported signed too, to expose winner's-curse drag.
  - efficiency: fraction of auctions won by the highest-true-value buyer.
  - price dispersion across repetitions (proxy for OTC's poor price discovery).

Pure standard library + a tiny bit of statistics; no third-party deps.
"""

import csv
import math
import random
import statistics

# ----------------------------------------------------------------------------
# CALIBRATION (from Freddie Mac SFLLD pool-level results, 2005-2008 vintages)
# ----------------------------------------------------------------------------
D_MEAN = 0.063     # mean pool default rate across pools
D_SD   = 0.044     # cross-pool sd of default rate  -> the common-value spread
LGD    = 0.41      # realized loss-given-default (loss rate / default rate)

# Private-value component: kept SMALL relative to common-value dispersion.
# Common-value value-spread is roughly LGD * D_SD = 0.41*0.044 ~ 0.018.
BETA   = 1.0       # scales c_i; with TAU below, private spread << common spread
TAU    = 0.003     # sd of private servicing-cost term (small)

SELLER_RESERVATION_FRAC = 0.5  # OTC: seller captures half the gains from trade

# ----------------------------------------------------------------------------
# Beta-distribution parameters matched to D_MEAN, D_SD (method of moments).
# ----------------------------------------------------------------------------
def beta_params(mean, sd):
    var = sd * sd
    # require var < mean(1-mean) for a valid Beta
    max_var = mean * (1 - mean)
    if var >= max_var:
        var = 0.99 * max_var
    common = mean * (1 - mean) / var - 1
    a = mean * common
    b = (1 - mean) * common
    return a, b

A_BETA, B_BETA = beta_params(D_MEAN, D_SD)


def draw_default_rate(rng):
    """Draw a pool true default rate D ~ Beta(a,b) in (0,1)."""
    return rng.betavariate(A_BETA, B_BETA)


def true_value(D):
    """Per-dollar value of the pool given true default rate D."""
    return 1.0 - LGD * D


# ----------------------------------------------------------------------------
# Bidder behavior
# ----------------------------------------------------------------------------
def naive_estimate(s_i, prior_mean=D_MEAN, prior_var=D_SD**2, sigma=0.0):
    """Posterior mean of D given one Gaussian signal s_i about D with a
    Gaussian(prior_mean, prior_var) prior and signal noise sigma.
    (Gaussian approximation to the Beta prior for tractable updating.)"""
    if sigma <= 0:
        return s_i
    obs_prec = 1.0 / (sigma * sigma)
    prior_prec = 1.0 / prior_var
    post = (prior_prec * prior_mean + obs_prec * s_i) / (prior_prec + obs_prec)
    return post


def shaded_bid(s_i, c_i, n, sigma):
    """Winner's-curse-aware bid in a common-value uniform-price auction.

    A buyer who wins learns her signal was the highest of n. Conditional on
    'my signal is the max', the expected underlying D is higher than the naive
    posterior (bad news: higher D -> lower value). We approximate the order-
    statistic correction: condition the signal estimate on being the max of n
    draws, which shifts the implied D upward by an amount that grows with the
    spread of signals and with n.

    Concretely: estimate D from s_i, then add a winner's-curse premium to the
    implied default rate proportional to the expected gap between the max signal
    and a typical signal, E[max - typical], which for n iid draws scales with
    sigma * a(n). Higher implied D lowers the value, i.e. shades the bid down.
    """
    D_hat = naive_estimate(s_i, sigma=sigma)
    # Expected normal order-statistic gap between the max of n and the mean,
    # in units of sigma. Closed-form-ish approximation (Blom): for the max of n,
    # E[max] ~ mu + sigma * Phi^{-1}((n - 0.375)/(n + 0.25)).
    if n <= 1:
        wc_gap_sd = 0.0
    else:
        p = (n - 0.375) / (n + 0.25)
        wc_gap_sd = inv_normal_cdf(p)
    # The winning signal is biased high by ~ wc_gap_sd * sigma relative to a
    # random signal; translate that into an upward correction on implied D.
    D_corrected = D_hat + wc_gap_sd * sigma * 0.5  # 0.5: signal->posterior shrink
    D_corrected = min(max(D_corrected, 0.0), 1.0)
    v_est = (1.0 - LGD * D_corrected) + BETA * c_i
    return v_est


def inv_normal_cdf(p):
    """Acklam's rational approximation to the inverse normal CDF."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ----------------------------------------------------------------------------
# One auction realization
# ----------------------------------------------------------------------------
def run_one(n, sigma, rng):
    D = draw_default_rate(rng)
    V = true_value(D)                      # the common-value fundamental

    signals = [D + rng.gauss(0, sigma) for _ in range(n)]
    privs   = [rng.gauss(0, TAU) for _ in range(n)]
    true_vals = [(1.0 - LGD * D) + BETA * c for c in privs]

    # ---- Uniform-price auction with shaded bids ----
    bids = [shaded_bid(signals[i], privs[i], n, sigma) for i in range(n)]
    # winner = highest bid; uniform price = 2nd-highest bid (uniform-price for
    # a single unit reduces to the second-price clearing level).
    order = sorted(range(n), key=lambda i: bids[i], reverse=True)
    winner = order[0]
    clearing_price = bids[order[1]] if n >= 2 else bids[order[0]]

    auction_gap = clearing_price - V               # signed
    # efficiency: did the winner land in the TOP of the true-value distribution?
    # (With a tiny private component, "exact best" is near-noise; what matters
    # is whether the mechanism awards the pool to a high-value buyer.)
    ranked = sorted(range(n), key=lambda i: true_vals[i], reverse=True)
    top_decile_cut = max(1, n // 10)
    winner_rank = ranked.index(winner)
    eff = 1.0 if winner_rank < top_decile_cut else 0.0
    # welfare ratio: winner's true value relative to best/worst spread
    best_v = true_vals[ranked[0]]
    worst_v = true_vals[ranked[-1]]
    welfare_ratio = ((true_vals[winner] - worst_v) / (best_v - worst_v)
                     if best_v > worst_v else 1.0)

    # ---- OTC benchmark: one random buyer negotiates with the seller ----
    j = rng.randrange(n)
    buyer_est = naive_estimate(signals[j], sigma=sigma)
    buyer_val = (1.0 - LGD * buyer_est) + BETA * privs[j]
    # seller reservation ~ prior expected value (no pool-specific info)
    seller_res = true_value(D_MEAN)
    if buyer_val >= seller_res:
        otc_price = seller_res + SELLER_RESERVATION_FRAC * (buyer_val - seller_res)
    else:
        otc_price = buyer_val  # no trade region; record buyer's value as the mark
    otc_gap = otc_price - V

    return auction_gap, otc_gap, eff, clearing_price, otc_price, welfare_ratio


def run_cell(n, sigma, reps, seed):
    rng = random.Random(seed)
    a_gaps, o_gaps, effs, a_prices, o_prices, welf = [], [], [], [], [], []
    for _ in range(reps):
        ag, og, e, ap, op, wr = run_one(n, sigma, rng)
        a_gaps.append(ag); o_gaps.append(og); effs.append(e)
        a_prices.append(ap); o_prices.append(op); welf.append(wr)
    return {
        "n": n, "sigma": sigma,
        "auction_abs_gap": statistics.mean(abs(x) for x in a_gaps),
        "auction_signed_gap": statistics.mean(a_gaps),
        "otc_abs_gap": statistics.mean(abs(x) for x in o_gaps),
        "top_decile_hit": statistics.mean(effs),
        "welfare_ratio": statistics.mean(welf),
        "auction_price_sd": statistics.pstdev(a_prices),
        "otc_price_sd": statistics.pstdev(o_prices),
    }


def shaded_bid_multiunit(s_i, c_i, n, k, sigma):
    """Winner's-curse-aware bid for a k-unit uniform-price auction.

    In a single-unit auction a winner conditions on having the MAX signal. With
    k units, the marginal (price-setting) bidder conditions on sitting at the
    k-th position, i.e. at the (1 - k/n) quantile of signals, not the top. The
    pivotal-signal bias is therefore the order-statistic gap at that quantile,
    which SHRINKS as k/n rises. This is the mechanism behind Pesendorfer-
    Swinkels convergence under double largeness."""
    D_hat = naive_estimate(s_i, sigma=sigma)
    if n <= 1:
        q_gap = 0.0
    else:
        # marginal bidder sits at quantile (1 - k/n) from the bottom
        q = max(min(1.0 - k / n, 1 - 1e-6), 1e-6)
        q_gap = inv_normal_cdf(q)
    D_corrected = D_hat + q_gap * sigma * 0.5
    D_corrected = min(max(D_corrected, 0.0), 1.0)
    return (1.0 - LGD * D_corrected) + BETA * c_i


def run_calibration_scenario(d_mean, d_sd, lgd, n_grid, sigma, reps, seed):
    """Re-run the single-unit experiment under an ALTERNATIVE calibration of the
    common-value distribution, to test whether the qualitative pattern depends
    on the (crisis-era) baseline numbers. Temporarily overrides the module-level
    Beta parameters and LGD."""
    global A_BETA, B_BETA, LGD
    a_save, b_save, lgd_save = A_BETA, B_BETA, LGD
    A_BETA, B_BETA = beta_params(d_mean, d_sd)
    LGD = lgd
    out = []
    s = seed
    for n in n_grid:
        s += 1
        out.append(run_cell(n, sigma, reps, s))
    A_BETA, B_BETA, LGD = a_save, b_save, lgd_save
    return out


def run_beta_sensitivity(beta_vals, n, sigma, reps, seed):
    """Vary the private-value weight beta to confirm the common value dominates
    and that conclusions are not driven by the private component."""
    global BETA
    save = BETA
    out = []
    s = seed
    for bv in beta_vals:
        BETA = bv
        s += 1
        out.append({"beta": bv, **run_cell(n, sigma, reps, s)})
    BETA = save
    return out


def run_multiunit(n, k, sigma, reps, seed):
    """Double-largeness experiment: k identical units (pools) auctioned to n
    bidders, each wanting one unit. Uniform price = the (k+1)-th highest bid.
    As both n and k grow with k/n fixed, Pesendorfer-Swinkels predicts the
    clearing price converges to the true common value V."""
    rng = random.Random(seed)
    abs_gaps = []
    for _ in range(reps):
        D = draw_default_rate(rng)
        V = true_value(D)
        signals = [D + rng.gauss(0, sigma) for _ in range(n)]
        privs = [rng.gauss(0, TAU) for _ in range(n)]
        bids = [shaded_bid_multiunit(signals[i], privs[i], n, k, sigma)
                for i in range(n)]
        bids.sort(reverse=True)
        # uniform clearing price for k units = first losing bid = (k+1)-th
        price = bids[k] if k < n else bids[-1]
        abs_gaps.append(abs(price - V))
    return {"n": n, "k": k, "sigma": sigma,
            "auction_abs_gap": statistics.mean(abs_gaps)}


def main():
    REPS = 5000
    N_GRID = [2, 5, 10, 25, 50, 100]
    SIGMA_GRID = [0.02, 0.04, 0.08]   # signal noise (sd of buyer signal about D)

    rows = []
    seed = 12345
    for sigma in SIGMA_GRID:
        for n in N_GRID:
            seed += 1
            rows.append(run_cell(n, sigma, REPS, seed))

    # Write full grid
    with open("sim_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # Double-largeness: scale n and k together (k = 30% of n), sigma fixed
    mu_rows = []
    for n in [10, 20, 50, 100, 200, 400]:
        seed += 1
        k = max(1, int(0.30 * n))
        mu_rows.append(run_multiunit(n, k, 0.04, REPS, seed))
    with open("sim_multiunit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mu_rows[0].keys()))
        w.writeheader()
        for r in mu_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    # Headline table at the middle sigma, for the paper
    mid = 0.04
    headline = [r for r in rows if abs(r["sigma"] - mid) < 1e-9]

    print(f"Calibration: D_mean={D_MEAN}, D_sd={D_SD}, LGD={LGD}")
    print(f"Beta(a,b) = ({A_BETA:.3f}, {B_BETA:.3f})   reps/cell={REPS}\n")
    print(f"HEADLINE (single unit, signal sd sigma = {mid}):")
    print(f"{'n':>4} | {'auction|gap|':>12} | {'OTC|gap|':>9} | "
          f"{'top-decile':>10} | {'welfare':>8} | {'auc price sd':>12} | "
          f"{'OTC price sd':>12}")
    print("-" * 88)
    for r in headline:
        print(f"{r['n']:>4} | {r['auction_abs_gap']:>12.5f} | "
              f"{r['otc_abs_gap']:>9.5f} | {r['top_decile_hit']:>10.3f} | "
              f"{r['welfare_ratio']:>8.3f} | {r['auction_price_sd']:>12.5f} | "
              f"{r['otc_price_sd']:>12.5f}")

    print("\nSigned auction gap (negative => winner's-curse shading below value):")
    for r in headline:
        print(f"  n={r['n']:>4}  signed_gap = {r['auction_signed_gap']:+.5f}")

    print("\nDOUBLE-LARGENESS (k = 0.30*n units, sigma = 0.04):")
    print(f"{'n':>5} | {'k':>4} | {'auction |gap|':>13}")
    print("-" * 30)
    for r in mu_rows:
        print(f"{r['n']:>5} | {r['k']:>4} | {r['auction_abs_gap']:>13.5f}")

    # --- Robustness 1: calm-vintage calibration scenario ---
    # Illustrative low-dispersion calibration (e.g. a benign origination year):
    # much lower mean default and tighter spread than the 2005-08 baseline.
    calm = run_calibration_scenario(0.010, 0.008, 0.30, N_GRID, 0.04, REPS, 9000)
    print("\nROBUSTNESS 1 -- CALM-VINTAGE CALIBRATION (D_mean=0.010, D_sd=0.008):")
    print(f"{'n':>4} | {'auction|gap|':>12} | {'OTC|gap|':>9} | {'signed':>9}")
    print("-" * 42)
    for r in calm:
        print(f"{r['n']:>4} | {r['auction_abs_gap']:>12.5f} | "
              f"{r['otc_abs_gap']:>9.5f} | {r['auction_signed_gap']:>+9.5f}")

    # --- Robustness 2: private-value weight sensitivity ---
    betas = run_beta_sensitivity([0.0, 1.0, 3.0, 6.0], 10, 0.04, REPS, 9100)
    print("\nROBUSTNESS 2 -- PRIVATE-VALUE WEIGHT (n=10, sigma=0.04):")
    print(f"{'beta':>5} | {'auction|gap|':>12} | {'welfare':>8}")
    print("-" * 32)
    for r in betas:
        print(f"{r['beta']:>5.1f} | {r['auction_abs_gap']:>12.5f} | "
              f"{r['welfare_ratio']:>8.3f}")

    # write robustness CSVs
    with open("sim_robust_calm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(calm[0].keys()))
        w.writeheader()
        for r in calm:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    with open("sim_robust_beta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(betas[0].keys()))
        w.writeheader()
        for r in betas:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("\nWrote sim_results.csv, sim_multiunit.csv, sim_robust_calm.csv, sim_robust_beta.csv.")


if __name__ == "__main__":
    main()
