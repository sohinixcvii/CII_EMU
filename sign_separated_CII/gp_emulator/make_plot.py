#!/usr/bin/env python
"""Re-generate corner plot from saved chain (pk202.out)."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math

# Load chain — single-column file loads as 1D array
mhmin_samples = np.loadtxt('pk202.out')   # (40000,)
chain = mhmin_samples[:, np.newaxis]      # (40000, 1) for reshape later

# True value
DATA_DIR = '../data/'
params = np.loadtxt(DATA_DIR + 'params_t')
idx = 202
true_mhmin = params[idx, 0]

q16, q50, q84 = np.percentile(mhmin_samples, [16, 50, 84])
n_walkers = 4

print(f"Chain shape : {chain.shape}")
print(f"True Mhmin  : {true_mhmin:.6f}")
print(f"Median      : {q50:.6f}")
print(f"Mean        : {mhmin_samples.mean():.6f}")
print(f"Std         : {mhmin_samples.std():.6f}")
print(f"68% CI      : [{q16:.6f}, {q84:.6f}]")
print(f"Unique vals : {len(np.unique(mhmin_samples))}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# --- Left panel: posterior histogram ---
ax = axes[0]
ax.hist(mhmin_samples, bins=80, density=True, color='C0', alpha=0.5, label='samples')

xgrid = np.linspace(mhmin_samples.min() - 0.01, mhmin_samples.max() + 0.01, 500)
kde   = gaussian_kde(mhmin_samples, bw_method=0.05)
ax.plot(xgrid, kde(xgrid), 'C0', lw=2, label='KDE')

ax.axvline(true_mhmin, color='C1', lw=2, ls='--', label=f'truth = {true_mhmin:.4f}')
ax.axvline(q50,        color='k',  lw=1.5, ls='-',  label=f'median = {q50:.4f}')
ax.axvspan(q16, q84, alpha=0.15, color='C0',
           label=f'68% CI [{q16:.4f}, {q84:.4f}]')

ax.set_xlabel(r'$M_{\rm h,min}$', fontsize=14)
ax.set_ylabel('Probability density', fontsize=13)
ax.set_title(r'Posterior: $M_{\rm h,min}$', fontsize=13)
ax.legend(fontsize=10)

# --- Right panel: trace plot ---
ax2 = axes[1]
trace = mhmin_samples.reshape(-1, n_walkers)   # (10000, 4)
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

fig.savefig('corner_plot.png', dpi=150, bbox_inches='tight')
print("\nCorner plot saved → corner_plot.png")
