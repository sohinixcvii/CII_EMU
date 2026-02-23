# CII Emulator — Instruction Manual

## Overview

This project is an ANN-based emulator for the CII/21cm power spectrum. Given a set of
cosmological/astrophysical parameters, it predicts the dimensionless power spectrum
Δ²(k) across six k-bins. The workflow has three stages:

```
1. Hyperparameter optimisation  →  Hyperparameter_tuning.py
2. Training the emulator        →  CII_emu.py
3. MCMC parameter inference     →  MCMC.py
```

Stages 1 and 2 are independent (you can skip HPO and use the defaults in `ann_input.py`).
Stage 3 requires a trained model from Stage 2.

---

## 1. Environment Setup

Create and activate the conda environment once:

```bash
conda create -n cii_emu python=3.10 -y
conda activate cii_emu
pip install "tensorflow==2.13.0" "keras-tuner" "scikit-learn" \
            "numpy<2" "matplotlib" "emcee>=3.1" "tqdm"
```

> **Note:** `cosmoHammer` has been removed. The MCMC stage now uses
> `emcee 3.x` directly, which is actively maintained and has no external
> MPI dependencies for single-machine use. `tqdm` is optional but
> enables progress bars during sampling.

All subsequent commands assume the environment is active:

```bash
conda activate cii_emu
cd /path/to/sign_separated_CII
```

---

## 2. Required Data Files

All data files live in the `data/` subdirectory.

| File | Description |
|------|-------------|
| `data/Npk.txt` | Raw power spectra (columns 2–7 used); one row per parameter sample |
| `data/k.txt` | k-bin centres (6 values) |
| `data/nbins.txt` | Number of k-modes per bin (used in covariance) |
| `data/params_t` | Parameter samples used for training (`Mhmin`, `alpha`) |
| `data/params_LH.txt` | Latin Hypercube parameter samples (used by HPO only) |
| `data/cii_er` | Observational errors (used by MCMC likelihood) |

Do not rename these files; paths are set in `ann_input.py`.

---

## 3. Configuration — `ann_input.py`

This is the single file to edit before running anything. Key settings:

```python
# Data paths
path = 'data/'

# Train/test split
test_frac = 0.1       # 10% held out for testing

# Validation fraction (of training set)
val_frac = 0.2

# Best hyperparameters (from HPO or manual tuning)
layers        = 32
neurons       = 144
dropout       = True
dropout_rate  = 0.1
epochs        = 1000   # max epochs (EarlyStopping will cut this short)
batch         = 4
learning_rate = 9.9999999e-05
activation    = 'relu'
```

Change `layers`, `neurons`, `learning_rate`, etc. here to use different hyperparameters
without touching the training script.

---

## 4. Stage 1 — Hyperparameter Optimisation (optional)

Run this to search for the best network architecture. Results are printed and saved
to `tuning_results/`.

```bash
python Hyperparameter_tuning.py
```

**What it does:**
- Loads `Npk.txt`, `k.txt`, `nbins.txt`, `params_LH.txt`
- Converts raw power spectra to dimensionless Δ²(k) and applies log-scaling
- Splits data: 90% train / 10% test, then 80% train / 20% validation
- Runs Bayesian optimisation (5 trials, 3 executions each) by default
- Prints a results summary at the end

**Tuner choice** — edit `ann_input.py`:
```python
tuner_choice = 'BayesOpt'   # or 'HyperBand'
```

**Output:**
```
tuning_results/test_mod/     ← trial checkpoints and oracle
```

After completion, read off the best `layers`, `neurons`, `dropout_rate`, `lr`,
`activation` from the printed summary and copy them into `ann_input.py` before
proceeding to Stage 2.

---

## 5. Stage 2 — Training the Emulator

```bash
python CII_emu.py
```

**What it does:**
1. Loads `Npk.txt`, `k.txt`, `nbins.txt`, `params_t`
2. Computes dimensionless power spectrum: Δ²(k) = k³ · P(k) / (2π²)
3. Applies log-scaling: pk = log(Δ²) / 10
4. Splits 90/10 train/test (no shuffle, `random_state=10`)
5. Saves the test set immediately (`pk_test`, `params_test`)
6. Builds and trains the ANN using hyperparameters from `ann_input.py`
   - EarlyStopping: patience=20, monitors training loss, min_delta=1e-4
   - ModelCheckpoint: saves best model to `cii_bestmodel/`
7. Prints training and test accuracy/loss
8. Inverse-transforms predictions and computes mean percentage error
9. Saves predictions to `predictions`

**Expected output (console):**
```
Power-spectrum test set saved to file.
Parameters test set saved to file.
Epoch 1: loss improved from inf to X.XXXXX, saving model to cii_bestmodel
...
Training Accuracy is:  XX.XX   Testing accuracy is:  XX.XX
Training loss is:  X.XX        Testing loss is:  X.XX
Mean percentage error:  XX.XX
Predictions saved to file!
```

**Output files:**

| File | Contents |
|------|----------|
| `cii_bestmodel/` | Best model weights (SavedModel format) |
| `cii_model.h5` | Final model after all epochs (HDF5) |
| `cii_history.npy` | Training history dict (loss, RMSE per epoch) |
| `pk_test` | Test-set power spectra (tab-separated, one row per sample) |
| `params_test` | Test-set parameters (tab-separated) |
| `predictions` | Model predictions on test set (tab-separated) |

---

## 6. Stage 3 — MCMC Parameter Inference

`MCMC.py` uses **emcee** (affine-invariant ensemble sampler). It requires
`cii_model.h5` from Stage 2 plus `data/params_t`, `data/k.txt`, `data/Npk.txt`,
and `data/nbins.txt`.

```bash
python MCMC.py
```

When prompted:
```
Enter number of samples: 1000
```

Enter the number of production steps per walker. Burn-in is automatically 10% of
that number.

**What it does:**
1. Loads and preprocesses the power spectrum data (same pipeline as Stage 2)
2. For parameter index `idx=202` (hardcoded), uses the corresponding log-scaled
   power spectrum `fn_t[202]` as the "observed" data
3. Builds a diagonal Gaussian covariance from `data/nbins.txt`:
   `cov_ii = |data_i²| / nbins_i + |data_i / √nbins_i|`
4. Runs `emcee.EnsembleSampler` with `vectorize=True` so all walkers are
   batched into a single `model.predict()` call per step:
   - 4 walkers (matches original walker_ratio=2 × 2 params)
   - burn-in = 10% of samples (discarded, chain reset)
   - Free parameter: `Mhmin` ∈ (−0.4328, +0.4328), uniform prior
   - Fixed parameter: `alpha` = 0.395 (never sampled)
5. Saves the flattened chain and prints the mean acceptance fraction

**Configuration constants** (edit in `MCMC.py` to change):
```python
MHMIN_BOUNDS = (-0.4328129267019357, 0.4328129267019357)  # uniform prior limits
ALPHA_FIXED  = 0.395    # held fixed throughout
MODEL_PATH   = 'cii_model.h5'
```

**Output files** (written to `sign_separated_CII/`):

| File | Contents |
|------|----------|
| `pk202.out` | Flattened posterior samples, shape `(n_walkers × n_samples, 1)` |

The number `202` comes from `idx = 202` in `MCMC.py`.
The old CosmoHammer burnin/prob/state output files are no longer produced; burn-in
samples are discarded by `sampler.reset()` before the production run.

---

## 7. Evaluating Results

### 7.1 Emulator accuracy (Stage 2)

The console prints three metrics immediately after training:

- **RMSE** (`Training/Testing Accuracy`): root-mean-square error on the
  log-scaled power spectrum (×100 = %)
- **MSE loss** (`Training/Testing loss`): mean-squared error (×100 = %)
- **Mean percentage error**: computed on the inverse-transformed (physical) Δ²(k)

Lower is better. A mean percentage error below ~5% is typically acceptable.

To inspect the training curve:

```python
import numpy as np
import matplotlib.pyplot as plt

h = np.load('cii_history.npy', allow_pickle=True).item()
plt.plot(h['loss'], label='train loss')
plt.plot(h['root_mean_squared_error'], label='train RMSE')
plt.xlabel('Epoch'); plt.legend(); plt.show()
```

To compare predictions vs truth sample-by-sample:

```python
import numpy as np

pred    = np.loadtxt('predictions')   # shape (N_test, 6)
pk_true = np.loadtxt('pk_test')       # shape (N_test, 6)

err = (pred - pk_true) / pk_true
print("Per-k mean % error:", 100 * err.mean(axis=0))
print("Overall mean % error:", 100 * err.mean())
```

### 7.2 MCMC posterior (Stage 3)

Load and inspect the chain:

```python
import numpy as np
import matplotlib.pyplot as plt

chain = np.loadtxt('pk202.out')   # shape (n_walkers * n_samples, 1)

# Mhmin marginal posterior
plt.hist(chain[:, 0], bins=50)
plt.xlabel('Mhmin'); plt.ylabel('Counts')
plt.title('Posterior on Mhmin')
plt.show()

print("Posterior mean:", chain[:, 0].mean())
print("Posterior std:",  chain[:, 0].std())
```

To overlay the truth value:

```python
params_test = np.loadtxt('data/params_t')
truth = params_test[202]
plt.axvline(truth[0], color='k', ls='--', label='truth')
```

Check for chain convergence using the emcee autocorrelation time:

```python
import emcee, numpy as np

# Re-load as a 3-D chain (n_steps, n_walkers, n_dim) if you kept the sampler object,
# or estimate from the flat chain directly:
chain = np.loadtxt('pk202.out')   # (n_steps * n_walkers, 1)
print("Posterior std:", chain[:, 0].std())
# For a proper convergence check, inspect acceptance fraction printed at runtime.
# Aim for 0.2–0.5; if < 0.1 the walkers are stuck.
```

A well-converged run prints a mean acceptance fraction between 0.2 and 0.5. If it is
very low, increase `n_walkers` in the `run_mcmc()` call inside `MCMC.py`.

---

## 8. Typical Run Order

```bash
conda activate cii_emu
cd sign_separated_CII

# Optional: find best hyperparameters
python Hyperparameter_tuning.py
# → copy best params into ann_input.py

# Train the emulator
python CII_emu.py
# → inspect console output; check mean % error

# Run MCMC inference
python MCMC.py
# → enter number of samples when prompted (e.g. 1000 for a quick run, 5000+ for production)
# → inspect pk202.out
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ImportError: cannot import name ... from keras` | Wrong keras version | Ensure `tensorflow==2.13.0` is installed |
| `ModuleNotFoundError: No module named 'emcee'` | emcee not installed | `pip install "emcee>=3.1"` |
| `FileNotFoundError: data/Npk.txt` | Wrong working directory | Run scripts from inside `sign_separated_CII/` |
| `EOFError: EOF when reading a line` (MCMC) | stdin not attached | Run `python MCMC.py` directly in a terminal, not piped |
| Mean % error > 20% | Too few epochs or poor hyperparameters | Increase `epochs` in `ann_input.py` or re-run HPO |
| MCMC acceptance fraction < 0.1 | Walkers stuck; proposal scale too large | Increase `n_walkers` in `run_mcmc()` call inside `MCMC.py` |
| No progress bar shown during sampling | `tqdm` not installed | `pip install tqdm` |

---

## 10. Changelog

### 2026-02-23

**`MCMC.py` — replaced CosmoHammer with emcee**

`cosmoHammer` is no longer maintained. `MCMC.py` has been rewritten to use
`emcee 3.x` (the affine-invariant ensemble sampler that CosmoHammer itself
was built on top of).

Changes made:

- Removed all `cosmoHammer` imports and the `Core_Module` / `Likelihood_Module` /
  `RunMCMC` class structure that existed only to satisfy the CosmoHammer API
- Added `make_log_posterior()` which returns a **vectorized** log-posterior
  function: emcee passes all walker positions at once, so a single
  `model.predict(batch)` call handles every walker per step — much faster
  than the old one-at-a-time approach
- Replaced nested Python loops for computing Δ²(k) with NumPy broadcasting:
  `dpk = k**3 * npk / (2 * np.pi**2)`
- Covariance inversion now uses `np.diag(1/cov_diag)` instead of
  `np.linalg.inv` on a diagonal matrix
- Removed unused imports: `MinMaxScaler`, `matplotlib`, `math`, `Params`
- Changed `from ann_input import *` to explicit named imports
- Replaced `math.pi` with `np.pi`
- Moved prior bounds and fixed parameters to named module-level constants
  (`MHMIN_BOUNDS`, `ALPHA_FIXED`, `MODEL_PATH`)
- Removed all dead/commented-out code and Jupyter notebook `# In[N]:` tags
- Output file: only `pk{index}.out` (burn-in samples discarded via
  `sampler.reset()`; the old CosmoHammer `burnin.out`, `prob.out`, etc. are gone)
- `emcee` dependency added; `cosmoHammer` dependency removed

**`MANUAL.md`** updated to reflect all of the above.
