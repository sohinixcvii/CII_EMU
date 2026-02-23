#!/usr/bin/env python
"""
Run GP-MCMC with 10 000 production steps per walker (4 walkers = 40 000 samples
total) and produce a corner plot of the Mhmin posterior.

Run from inside gp_emulator/:
    python run_corner.py
"""

import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

from GP_MCMC import load_data, run_mcmc

# ---------------------------------------------------------------------------
# Run MCMC
# ---------------------------------------------------------------------------
fn_t, nbins, params_test = load_data()
idx = 202
true_mhmin = params_test[idx, 0]
print(f"Params[{idx}]: Mhmin={true_mhmin:.6f}, alpha={params_test[idx,1]:.6f}")

chain, sampler = run_mcmc(
    data=fn_t[idx],
    nbins=nbins,
    index=idx,
    n_walkers=4,
    burnin_frac=0.1,
    n_samples=10_000,
)

# chain shape: (4 * 10000, 1)
print(f"\nChain shape: {chain.shape}")
print(f"Mhmin posterior:  mean={chain[:,0].mean():.6f}  std={chain[:,0].std():.6f}")
print(f"True Mhmin:       {true_mhmin:.6f}")

# ---------------------------------------------------------------------------
# Corner plot
# For a 1-parameter problem, corner.corner fails on truth lines (known issue).
# We build the figure manually: 1D marginal histogram + trace plot.
# ---------------------------------------------------------------------------
from scipy.stats import gaussian_kde

q16, q50, q84 = np.percentile(chain[:, 0], [16, 50, 84])
mhmin_samples  = chain[:, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# --- Left panel: marginal posterior histogram ---
ax = axes[0]
ax.hist(mhmin_samples, bins=80, density=True,
        color='C0', alpha=0.5, label='samples')

# KDE overlay
xgrid = np.linspace(mhmin_samples.min() - 0.01, mhmin_samples.max() + 0.01, 500)
kde   = gaussian_kde(mhmin_samples, bw_method=0.05)
ax.plot(xgrid, kde(xgrid), 'C0', lw=2, label='KDE')

# Truth and quantile lines
ax.axvline(true_mhmin, color='C1', lw=2, ls='--', label=f'truth = {true_mhmin:.4f}')
ax.axvline(q50,        color='k',  lw=1.5, ls='-',  label=f'median = {q50:.4f}')
ax.axvspan(q16, q84, alpha=0.15, color='C0', label=f'68% CI [{q16:.4f}, {q84:.4f}]')

ax.set_xlabel(r'$M_{\rm h,min}$', fontsize=14)
ax.set_ylabel('Probability density', fontsize=13)
ax.set_title(r'Posterior: $M_{\rm h,min}$', fontsize=13)
ax.legend(fontsize=10)

# --- Right panel: trace plot ---
ax2 = axes[1]
# Reshape to (n_steps, n_walkers) for trace display
n_walkers = 4
trace = mhmin_samples.reshape(-1, n_walkers)   # (n_steps, n_walkers)
for w in range(n_walkers):
    ax2.plot(trace[:, w], alpha=0.6, lw=0.6, label=f'walker {w}')
ax2.axhline(true_mhmin, color='C1', lw=1.5, ls='--', label='truth')
ax2.set_xlabel('Step', fontsize=13)
ax2.set_ylabel(r'$M_{\rm h,min}$', fontsize=13)
ax2.set_title('Trace plot', fontsize=13)
ax2.legend(fontsize=9, ncol=2)

fig.suptitle(
    f"GP-MCMC posterior  (idx={idx}, 4×10 000 samples)\n"
    f"True $M_{{\\rm h,min}}$ = {true_mhmin:.4f}  |  "
    f"Recovered = {q50:.4f}  (+{q84-q50:.4f} / -{q50-q16:.4f})",
    fontsize=12, y=1.02
)
fig.tight_layout()

out = 'corner_plot.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nCorner plot saved → {out}")

# ---------------------------------------------------------------------------
# Acceptance fraction summary
# ---------------------------------------------------------------------------
af = np.mean(sampler.acceptance_fraction)
print(f"Mean acceptance fraction: {af:.3f}  "
      f"({'OK' if 0.1 < af < 0.9 else 'WARNING: outside 0.1-0.9'})")
