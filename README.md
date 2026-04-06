# Tax-Aware Portfolio Optimization for Separately Managed Accounts (SMAs)

[![R](https://img.shields.io/badge/R-%3E%3D4.2-276DC3?style=flat-square&logo=r&logoColor=white)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Solver: ECOS_BB](https://img.shields.io/badge/Solver-ECOS__BB-blue?style=flat-square)](https://github.com/embotech/ecos)
[![Solver: OSQP](https://img.shields.io/badge/Solver-OSQP-orange?style=flat-square)](https://osqp.org/)
[![Course](https://img.shields.io/badge/Course-MATH%204100-red?style=flat-square)](https://www.wit.edu/)

> **MATH 4100 – Industrial Problems in Applied Mathematics**
> Wentworth Institute of Technology

A mixed-integer quadratic programming (MIQP) optimizer for tax-aware portfolio construction in separately managed accounts, featuring Bayes-Stein return shrinkage, IRC Section 1256 futures tax treatment, Monte Carlo simulation, and an efficient frontier sweep across S&P 500 equities and futures contracts.

---

## Table of Contents

- [Background: SMAs and Tax-Aware Optimization](#background-smas-and-tax-aware-optimization)
- [Mathematical Formulation](#mathematical-formulation)
- [Tax Model and Linearization](#tax-model-and-linearization)
- [Bayes-Stein Return Shrinkage](#bayes-stein-return-shrinkage)
- [IRC Section 1256 Futures Tax Treatment](#irc-section-1256-futures-tax-treatment)
- [Optimization Modes](#optimization-modes)
- [Additional Features](#additional-features)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Results](#sample-results)
- [File Structure](#file-structure)
- [References](#references)
- [License](#license)

---

## Background: SMAs and Tax-Aware Optimization

### What Is a Separately Managed Account?

A **separately managed account (SMA)** is an individually owned investment portfolio managed on behalf of a single investor by a professional portfolio manager. Unlike mutual funds or ETFs — in which thousands of investors pool capital and share identical portfolios — an SMA holds securities directly in the investor's own name. This structural distinction has a critical consequence: **every trading decision can be personalized to the investor's specific tax situation**.

In a mutual fund, an investor may receive a capital gains distribution even if they never sold a single share, simply because other investors redeemed shares and forced the fund to liquidate positions. In an SMA, no such forced liquidation occurs. The optimizer can consider the investor's cost basis in every position, their marginal tax rate, and their embedded unrealized gains before deciding whether to sell.

### Why Tax-Aware Optimization Matters

The canonical Markowitz mean-variance framework optimizes pre-tax returns against portfolio variance. For taxable investors — particularly high-net-worth individuals in top marginal brackets — this approach systematically overestimates the value of rebalancing. Selling a position that has appreciated substantially triggers a capital gains liability, reducing net-of-tax wealth. The optimal pre-tax portfolio and the optimal after-tax portfolio can differ materially.

This project embeds the tax cost of rebalancing directly into the optimization objective and constraints, so the solver naturally avoids selling high-gain positions unless doing so produces a sufficient improvement in risk-adjusted after-tax return to justify the tax drag.

---

## Mathematical Formulation

### Decision Variables

| Symbol | Description |
|:---|:---|
| $w_s \in \mathbb{R}^{n_s}$ | Portfolio weight vector for S&P 500 equities |
| $w_f \in \mathbb{R}^{n_f}$ | Portfolio weight vector for futures contracts |
| $z_s \in \{0,1\}^{n_s}$ | Binary asset selection indicators for stocks |
| $z_f \in \{0,1\}^{n_f}$ | Binary asset selection indicators for futures |
| $s_i \geq 0$ | Auxiliary variable linearizing $\max(0,\, w_{\text{cur},i} - w_i)$ |

### Core MIQP Objective

Minimize total portfolio variance across both asset classes:

$$\min_{w_s,\, w_f,\, z_s,\, z_f,\, s} \quad w_s^\top \Sigma_s\, w_s \;+\; w_f^\top \Sigma_f\, w_f$$

### Constraints

**Budget constraints** — weights must sum to their respective allocation fractions, which together sum to one:

$$\sum_i w_{s,i} = \alpha_s, \qquad \sum_j w_{f,j} = \alpha_f, \qquad \alpha_s + \alpha_f = 1$$

**After-tax return floor (Mode 1):**

$$R_s^\top w_s \;+\; R_f^\top w_f \;-\; \tau \sum_i g_i s_i \;\geq\; \mu_{\text{target}}$$

**Variance cap (Mode 2):**

$$w_s^\top \Sigma_s\, w_s \;+\; w_f^\top \Sigma_f\, w_f \;\leq\; \sigma^2_{\text{max}}$$

**Cardinality constraints** — exactly $k$ assets selected per class:

$$\sum_i z_{s,i} = k_s, \qquad \sum_j z_{f,j} = k_f$$

**Linking constraints** — weights activate only for selected assets:

$$w_{\min}\, z_i \;\leq\; w_i \;\leq\; z_i, \qquad \forall\, i$$

**Tax linearization constraints:**

$$s_i \geq w_{\text{cur},i} - w_i, \qquad s_i \geq 0, \qquad \forall\, i$$

**Non-negativity:**

$$w_i \geq 0, \qquad \forall\, i$$

### Parameter Definitions

| Symbol | Definition |
|:---|:---|
| $\Sigma_s,\, \Sigma_f$ | Annualized covariance matrices for stocks and futures |
| $R_s,\, R_f$ | Expected return vectors (post Bayes-Stein shrinkage) |
| $\tau$ | Investor's long-term capital gains tax rate |
| $g_i = \max(0,\, 1 - b_i)$ | Unrealized gain fraction, where $b_i$ is cost basis as a fraction of current price |
| $w_{\text{cur},i}$ | Investor's current weight in stock $i$ before rebalancing |
| $\mu_{\text{target}}$ | Investor-specified after-tax return floor |
| $\sigma^2_{\text{max}}$ | Variance cap derived from investor's risk tolerance |

---

## Tax Model and Linearization

### Capital Gains Tax Drag

When the optimizer rebalances from a current weight $w_{\text{cur},i}$ to a new weight $w_i$, selling stock $i$ (i.e., $w_i < w_{\text{cur},i}$) triggers a realized capital gain. The after-tax cost of this rebalancing, normalized per unit of portfolio value, is:

$$\text{Tax drag}_i = \tau \cdot g_i \cdot \max\!\bigl(0,\; w_{\text{cur},i} - w_i\bigr)$$

where $g_i = \max(0, 1 - b_i)$ is the embedded unrealized gain fraction and $b_i$ is the cost basis as a fraction of the current market price. A position purchased at 70% of its current price has $b_i = 0.70$ and $g_i = 0.30$.

The term $\max(0, w_{\text{cur},i} - w_i)$ is convex in $w_i$ but not linear — it cannot be passed directly to a quadratic solver. The standard technique is **auxiliary variable linearization**: introduce $s_i \geq 0$ with the constraint $s_i \geq w_{\text{cur},i} - w_i$. At optimality, $s_i = \max(0, w_{\text{cur},i} - w_i)$ exactly, because the objective includes $-\tau g_i s_i$ (minimizing tax drag means minimizing $s_i$), so the solver has no incentive to inflate $s_i$ beyond the binding constraint.

$$s_i^\star = \max\!\bigl(0,\; w_{\text{cur},i} - w_i\bigr) \qquad \text{at optimality}$$

This substitution keeps the problem in the **MIQP class** without introducing any new solver or approximation.

### After-Tax Expected Return

The aggregate after-tax expected return of the portfolio is:

$$\mu_{\text{AT}} = R_s^\top w_s + R_f^\top w_f - \tau \sum_i g_i s_i^\star$$

The optimizer replaces the pre-tax return constraint with this after-tax expression, so the solver natively avoids selling appreciated positions — not through a penalty heuristic, but because doing so is explicitly more expensive in the objective.

---

## Bayes-Stein Return Shrinkage

Historical mean returns are notoriously noisy estimators of true expected returns. Plugging raw sample means directly into a mean-variance optimizer often produces extreme, unstable allocations that overfit to historical anomalies. **Bayes-Stein grand mean shrinkage** (Jorion 1985) addresses this by pulling each asset's estimated return toward the cross-sectional grand mean:

$$\hat{\mu}_i = (1 - \lambda)\,\mu_{\text{raw},i} + \lambda\,\bar{\mu}_{\text{grand}}$$

where the shrinkage intensity is:

$$\lambda = \frac{n}{n + T}$$

Here $n$ is the number of assets and $T$ is the number of return observations. With many assets and few observations (large $n/T$), $\lambda \to 1$ and returns are pulled strongly toward the grand mean. With abundant data (small $n/T$), $\lambda \to 0$ and raw estimates are used. This produces more conservative, stable return inputs and materially reduces in-sample overfitting.

**Empirical impact in this project** (representative run):

| Configuration | Pre-Tax Return | After-Tax Return | Sharpe Ratio |
|:---|:---:|:---:|:---:|
| Without Bayes-Stein | 35.16% | 30.89% | 1.385 |
| With Bayes-Stein | 17.38% | 14.32% | 0.620 |

The shrinkage-adjusted estimates are materially more conservative and better reflect realistic expectations for out-of-sample performance.

---

## IRC Section 1256 Futures Tax Treatment

Futures contracts held in U.S. taxable accounts are governed by **Internal Revenue Code § 1256** (26 U.S.C. § 1256), which mandates a fixed blended tax rate of **60% long-term / 40% short-term** capital gains treatment, regardless of actual holding period. Positions are marked to market at year-end, so unrealized gains are taxed annually.

Because the Section 1256 rate is **fixed by statute** and does not depend on the weight vector $w_f$, the futures tax drag is a constant dollar amount that does not affect the optimization. It is therefore excluded from the constraint set and reported separately in the output as a post-optimization accounting adjustment.

The effective blended Section 1256 rate for a taxpayer in the 37% ordinary bracket (23.8% LTCG rate including net investment income tax) would be approximately:

$$\tau_{1256} = 0.60 \times 0.238 + 0.40 \times 0.37 \approx 0.2908$$

This is reported in the output alongside the equity-side capital gains tax drag.

---

## Optimization Modes

### Mode 1 — Minimize Risk (Subject to After-Tax Return Floor)

The investor specifies a target after-tax return $\mu_{\text{target}}$. The optimizer minimizes total portfolio variance subject to the after-tax return constraint, cardinality constraints, and all tax linearization constraints.

**Use case:** An investor with a defined return objective (e.g., "I need 12% after taxes") who wants the lowest-risk portfolio that meets that target.

### Mode 2 — Maximize After-Tax Return (Subject to Variance Cap)

The investor selects a risk level — **Low**, **Medium**, or **High** — which maps to the 25th, 50th, or 75th percentile of individual asset return volatilities, respectively. The optimizer maximizes after-tax return subject to the variance cap.

**Use case:** An investor comfortable with a specified volatility budget who wants the highest after-tax return achievable within it.

---

## Additional Features

**Tax-Aware Efficient Frontier**
A 50-point sweep across after-tax return targets, solved with the OSQP solver (continuous relaxation), traces the risk–return frontier with tax drag embedded. The y-axis reports after-tax expected return, making this a true after-tax frontier rather than a pre-tax approximation.

**Covariance Risk Decomposition**
Eigenvalue decomposition of $\Sigma_s$ and $\Sigma_f$ separately. The top 8 principal components are reported for each, showing what fraction of total portfolio risk is attributable to systematic vs. idiosyncratic factors.

**Dollar P&L Comparison**
Side-by-side comparison of three portfolio constructions: 100% equities (tax-aware), 100% futures (Section 1256), and the combined allocation. Output includes gross return, tax drag in dollar terms, and net-of-tax P&L.

**Monte Carlo Simulation (5,000 Paths)**
Geometric Brownian motion with Itô correction over a 1-year horizon:

$$\ln S_T = \ln S_0 + \left(\mu_m - \tfrac{1}{2}\sigma_m^2\right)T + \sigma_m\sqrt{T}\,Z, \qquad Z \sim \mathcal{N}(0,1)$$

where $\mu_m$ is the after-tax expected portfolio return. Output includes fan charts, median path overlay, and final wealth histograms.

**Visualizations**
- Portfolio allocation pie chart (viridis color palette)
- Efficient frontier with optimum marked
- Monte Carlo fan chart with 5th/95th percentile bands
- Final wealth distribution histogram
- Principal component bar charts for covariance decomposition

---

## Installation

### Prerequisites

- R ≥ 4.2.0
- Internet connection (for package installation and market data retrieval)

### Step 1 — Install Required Packages

Run the setup script first. It installs all dependencies and performs test solves with both ECOS_BB and OSQP to verify solver availability before the main script executes:

```r
source("00_setup_packages.R")
```

Packages installed:

| Package | Purpose |
|:---|:---|
| `CVXR` | Disciplined convex optimization interface |
| `ECOSolveR` | ECOS_BB MIQP solver backend |
| `osqp` | OSQP QP solver (efficient frontier) |
| `quantmod` | Market data retrieval from Yahoo Finance |
| `BatchGetSymbols` | Batch S&P 500 equity price download |
| `viridis` | Colorblind-safe color palettes |
| `ggplot2` | Publication-quality plotting |
| `dplyr` | Data manipulation |
| `tidyr` | Data reshaping |
| `Matrix` | Sparse matrix support |

### Step 2 — Verify Solver Installation

The setup script runs test optimizations with both solvers. Expected output:

```
[SETUP] Testing ECOS_BB solver...   PASS
[SETUP] Testing OSQP solver...      PASS
[SETUP] All packages ready. Run 01_main_optimizer.R to begin.
```

---

## Usage

### Running the Optimizer

```r
source("01_main_optimizer.R")
```

The script will prompt for investor-specific inputs:

```
--- TAX-AWARE SMA OPTIMIZER ---
Enter stock allocation fraction (0–1): 0.75
Enter futures allocation fraction (0–1): 0.25
Enter capital gains tax rate (e.g., 0.20 for 20%): 0.20
Enter mean embedded gain fraction (e.g., 0.30): 0.30

Select optimization mode:
  [1] Minimize Risk (set after-tax return target)
  [2] Maximize After-Tax Return (set risk tolerance)
Mode: 1

Enter target after-tax return (e.g., 0.12 for 12%): 0.12

Enter number of stocks to select (max 100): 20
Enter number of futures to select (max 28): 10
```

### Optimization Modes

```r
# Mode 1: Minimize variance subject to after-tax return floor
# Set mode = 1 at prompt, then enter mu_target

# Mode 2: Maximize after-tax return subject to variance cap
# Set mode = 2 at prompt, then select risk level (Low / Medium / High)
```

### Output

Results are written to `results/` and printed to the console, including:

- Optimal weight vectors $w_s^\star$, $w_f^\star$
- Selected assets ($z_s^\star$, $z_f^\star$)
- Pre-tax and after-tax expected return
- Total tax drag in percentage and dollar terms
- Pre-tax and after-tax Sharpe ratios
- Monte Carlo fan chart and final wealth histogram
- Efficient frontier plot
- Principal component decomposition charts

---

## Sample Results

Results from a representative optimizer run with the following investor profile:

| Parameter | Value |
|:---|:---|
| Stock allocation ($\alpha_s$) | 75% |
| Futures allocation ($\alpha_f$) | 25% |
| Capital gains tax rate ($\tau$) | 20% |
| Mean embedded gain ($\bar{g}$) | 30% |
| Stocks selected ($k_s$) | 20 of 100 |
| Futures selected ($k_f$) | 10 of 28 |
| Optimization mode | Mode 1 (minimize risk) |

### Portfolio Performance

| Metric | Value |
|:---|:---|
| Pre-tax expected return | 36.94% |
| Total tax drag | −4.42% |
| **After-tax expected return** | **32.52%** |
| Pre-tax Sharpe ratio | 1.489 |
| After-tax Sharpe ratio | 1.301 |
| Sharpe reduction from tax drag | 0.188 |

### Bayes-Stein Shrinkage Impact

| Configuration | Pre-Tax Return | After-Tax Return | Sharpe |
|:---|:---:|:---:|:---:|
| Without shrinkage | 35.16% | 30.89% | 1.385 |
| **With shrinkage** | **17.38%** | **14.32%** | **0.620** |

Shrinkage produces substantially more conservative and realistic estimates — more appropriate for investor-facing output and stress testing.

---

## File Structure

```
tax-aware-portfolio-optimization/
│
├── 00_setup_packages.R        # Package installation and solver verification
├── 01_main_optimizer.R        # Main MIQP optimizer and all analytics
│
├── results/                   # Output plots and console logs
│   ├── efficient_frontier.png
│   ├── allocation_pie.png
│   ├── montecarlo_fan.png
│   ├── montecarlo_histogram.png
│   └── pca_decomposition.png
│
├── report/                    # Written project report
│   └── tax_aware_sma_report.pdf
│
├── presentation/              # Slide deck
│   └── tax_aware_sma_slides.pdf
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## References

1. **Markowitz, H.** (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.

2. **Jorion, P.** (1985). Bayes-Stein Estimation for Portfolio Analysis. *Journal of Financial and Quantitative Analysis*, 21(3), 279–292.

3. **Dammon, R. M., Spatt, C. S., & Zhang, H. H.** (2001). Optimal Asset Location and Allocation with Taxable and Tax-Deferred Investing. *Journal of Finance*, 56(3), 999–1037.

4. **Glasserman, P.** (2004). *Monte Carlo Methods in Financial Engineering*. Springer.

5. **Wilcox, J., Horvitz, J., & diBartolomeo, D.** (2006). *Investment Management for Taxable Private Investors*. CFA Institute Research Foundation.

6. **Internal Revenue Code § 1256**, 26 U.S.C. § 1256 — Treatment of certain foreign currency contracts; futures contracts.

7. **CVXR Development Team** (2020). CVXR: An R Package for Disciplined Convex Optimization. *Journal of Statistical Software*, 94(14).

8. **Domahidi, A., Chu, E., & Boyd, S.** (2013). ECOS: An SOCP Solver for Embedded Systems. *European Control Conference*.

9. **Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S.** (2020). OSQP: An Operator Splitting Solver for Quadratic Programs. *Mathematical Programming Computation*, 12, 637–672.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed for MATH 4100 – Industrial Problems in Applied Mathematics, Wentworth Institute of Technology.*
