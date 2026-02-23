# Architecture & Design Reference

This document explains how the project works end-to-end: the physics, the data,
the maths, every source file, and the design decisions behind each stage. It is
intended for someone picking the project up for the first time.

---

## 1. What this project does

The goal is **Bayesian parameter inference** for the CII/21 cm cross-power
spectrum. Concretely:

1. We have 2 167 simulated power spectra, each produced by a different value of
   the astrophysical parameter **Mhmin** (minimum halo mass).
2. We train an **emulator** — a fast surrogate model — that takes Mhmin as input
   and predicts the corresponding power spectrum at 6 wavenumber (k) bins.
3. We run **MCMC** (Markov Chain Monte Carlo) to infer the posterior distribution
   over Mhmin given one "observed" spectrum from the simulation suite.

---

## 2. The physics in one paragraph

**CII** is the 158 µm fine-structure emission line of singly-ionised carbon. It
is one of the brightest tracers of star-forming galaxies and can be mapped in
three dimensions via line-intensity mapping (LIM). **21 cm** is the hyperfine
spin-flip transition of neutral hydrogen, mapped by radio telescopes. The
**cross-power spectrum** P_CII×21cm(k) captures how the two signals are
spatially correlated as a function of Fourier wavenumber k. It is sensitive to
the halo occupation of CII emitters.

The model has two parameters:
| Symbol | Physical meaning | Status |
|--------|-----------------|--------|
| **Mhmin** | log₁₀ of the minimum halo mass (M☉) that hosts CII emission | **free** — inferred by MCMC |
| **alpha** | power-law slope of the CII luminosity–halo-mass relation | **fixed** at 0.395 across all 2 167 training samples |

Because alpha has zero variance in the training set, the emulator is effectively
a 1-dimensional mapping from Mhmin → power spectrum.

---

## 3. Data files (`data/`)

| File | Shape | Contents |
|------|-------|----------|
| `Npk.txt` | (2167, 10) | Raw simulated power spectra. **Only columns 2–7** (0-indexed) are used — 6 k-bins in the scientific range of interest. |
| `k.txt` | (6,) | Wavenumber bin centres in Mpc⁻¹: `[0.136, 0.244, 0.437, 0.781, 1.396, 2.495]` |
| `nbins.txt` | (6,) | Number of independent Fourier modes in each k-bin: `[302, 1853, 10478, 59876, 341274, 1949358]`. Rises steeply with k — used to set the noise level in the likelihood. |
| `params_t` | (2167, 2) | Parameter grid. Column 0: Mhmin ∈ [−0.4328, +0.4328]. Column 1: alpha = 0.395 (constant). |
| `params_LH.txt` | (N, 2) | Latin Hypercube samples — used only by the HPO script. |
| `cii_dpk` | text | Intermediate file written by `Hyperparameter_tuning.py` (not used downstream). |

---

## 4. The data-preprocessing pipeline

Every script (ANN and GP) applies the same three-step transform to go from the
raw simulation output to the quantity the model fits/predicts:

```
Npk[i, j]          raw power spectrum (units: Mpc³)
    ↓  × k[j]³ / (2π²)
dpk[i, j]          dimensionless power spectrum Δ²(k)  [unitless]
    ↓  log(·) / 10
pk[i, j]            log-scaled target   ≈ O(−0.3 … −0.1)
```

**Why Δ²(k) = k³ P(k) / (2π²)?**
This is the standard cosmological dimensionless power spectrum. It measures the
variance in density fluctuations per logarithmic k interval and is O(1) for the
scales of interest rather than spanning many orders of magnitude.

**Why log/10?**
Compresses the remaining dynamic range into a range of ≈ 0.2 centred near zero.
The ANN loss (MSE) and the GP kernel both work better in a compact, smooth space.

**Important caveat — exponential amplification:**
A prediction error ε in pk-space corresponds to a factor of exp(10ε) in
physical Δ²(k) space. Even a log-space error of 0.04 maps to a 49% physical
error. This is why the low-k bins, where the function is hardest to learn, show
the worst physical-space accuracy despite similar log-space errors.

---

## 5. Emulator approaches

Two independent emulators are implemented. They share the same input/output
convention but differ radically in method.

### 5.1 ANN emulator (`CII_emu.py` + `functions.py`)

```
Input:  params   shape (n, 2)   [Mhmin, alpha]
Output: pk       shape (n, 6)   log(dpk)/10 at 6 k-bins
```

Architecture (from `ann_input.py`):
- 1 input layer: Dense(2, relu)
- 32 hidden layers: Dense(144, relu)
- 1 Dropout(0.1) layer
- 1 output layer: Dense(6, relu)
- Loss: MSE in log-space. Optimizer: Adam(lr=1e-4)
- Training: up to 1 000 epochs, EarlyStopping(patience=20)

**Known problems:**
1. `shuffle=False` in the train/test split puts all high-Mhmin samples into the
   test set. In that region spectra are nearly identical (std ~ 10⁻¹⁵), so the
   split looks artificially good. With a random split the ANN achieves 44–53%
   mean abs % error.
2. 32 × 144 = 4 608 parameters per layer for a 1D input function — massive
   overparameterisation.
3. Exponential amplification makes log-space errors look small while physical
   errors are large.

### 5.2 GP emulator (`gp_emulator/GP_emu.py`)

```
Input:  X       shape (n, 1)   Mhmin only (StandardScaler normalised)
Output: pk[:,i] shape (n,)     log(dpk)/10 for k-bin i
```

One `GaussianProcessRegressor` is trained per k-bin (6 total). They are saved
together in `gp_models.joblib` as `{'scaler': StandardScaler, 'gps': [gp0…gp5]}`.

**Kernel:**
```
k(x, x') = C(1.0) × Matérn(ν=2.5, l=0.1) + WhiteKernel(1e-5)
```
- `ConstantKernel` — overall amplitude; fitted by marginal-likelihood optimisation
- `Matérn(ν=2.5)` — once mean-square differentiable; appropriate for smooth
  physical functions; standard in cosmological emulators
- `WhiteKernel` — absorbs numerical noise from the simulations
- `normalize_y=True` — each GP internally subtracts its mean and scales by std,
  handling the different output ranges across k-bins
- `n_restarts_optimizer=3` — re-runs hyperparameter optimisation from different
  initialisations to avoid local maxima in the log-marginal-likelihood

**Why the GP wins:**
Because alpha is constant, the effective input dimension is 1. With 1 950
densely-sampled training points on a smooth 1D function, the GP essentially
performs exact smooth interpolation. It achieves < 0.25% mean abs % error on all
k-bins vs 44–53% for the ANN.

**Measured accuracy (random 90/10 split, random_state=42):**

| k-bin | k (Mpc⁻¹) | mean \|%err\| | max \|%err\| |
|-------|-----------|--------------|------------|
| 0 | 0.136 | 0.25% | 6.3% |
| 1 | 0.244 | 0.22% | 5.3% |
| 2 | 0.437 | 0.20% | 5.5% |
| 3 | 0.781 | 0.19% | 5.5% |
| 4 | 1.396 | 0.18% | 5.4% |
| 5 | 2.495 | 0.17% | 5.5% |

---

## 6. MCMC inference

### 6.1 Sampler

Both `MCMC.py` (ANN) and `gp_emulator/GP_MCMC.py` (GP) use
**emcee** — the affine-invariant ensemble sampler. Configuration:

| Setting | Value | Rationale |
|---------|-------|-----------|
| `n_walkers` | 4 | Minimum for 1D (must be ≥ 2 × n_dim and even) |
| `n_dim` | 1 | Only Mhmin is free; alpha is fixed |
| burn-in | 10% of `n_samples` | Discarded via `sampler.reset()` |
| `vectorize=True` | — | All walkers batched into one predict call per step |

### 6.2 Prior

Flat (uniform) over:
```
Mhmin ∈ (−0.4328129267019357, +0.4328129267019357)
```
This interval corresponds to the range in `params_t`. Outside it the log-prior
returns −∞.

### 6.3 Likelihood

A diagonal Gaussian:
```
log L = −½ Σᵢ (pred_i − data_i)² / cov_ii
```

where `data` is one row of the log-scaled power spectrum (the "observed"
measurement), `pred` is the emulator prediction for a proposed Mhmin, and the
diagonal covariance is:
```
cov_ii = |data_i²| / nbins_i  +  |data_i / √nbins_i|
```

The first term models sample variance (cosmic variance); the second models
thermal/shot noise. Both `pred` and `data` are in **log(dpk)/10 space** — no
exp() transform is applied inside the likelihood.

### 6.4 Vectorisation

`emcee` passes all walker positions simultaneously as an array of shape
`(n_walkers, 1)`. The posterior function:

```python
# ANN version (MCMC.py)
model_th = model.predict(params_batch, verbose=0)   # one call for all walkers

# GP version (GP_MCMC.py)
preds = np.column_stack([gp.predict(X_scaled) for gp in gps])  # 6 GPs, batched
```

This is substantially faster than calling the emulator once per walker per step.

---

## 7. File reference

### Root directory

| File | Role |
|------|------|
| `ann_input.py` | **Single configuration file** for the ANN pipeline. Edit this to change data paths, hyperparameters, or train/test split fraction. Do not import it from `gp_emulator/` — it has a TensorFlow dependency. |
| `functions.py` | ANN helper library: `build_model()` (assembles and trains the Keras network), `save_to_file()` (writes predictions/test sets to disk), `history_plot()` (plots training curves). |
| `CII_emu.py` | **ANN training script.** Loads data, splits 90/10 (no shuffle), calls `fn.build_model()`, saves predictions. |
| `MCMC.py` | **ANN-based MCMC script.** Loads `cii_model.h5`, runs emcee, saves chain to `pk{idx}.out`. |
| `Hyperparameter_tuning.py` | Runs keras-tuner (Bayesian optimisation or HyperBand) to find the best ANN architecture. Outputs to `tuning_results/`. Run this before `CII_emu.py` if you want to search for better hyperparameters. |
| `Cross_emu.py` | Older, interactive version of the training script (uses `input()` for epochs/batch). Not part of the main pipeline; kept for reference. |
| `environment.yaml` | Conda environment spec. Creates the `cii_emu` environment with all dependencies. |
| `MANUAL.md` | Step-by-step user guide for running the pipeline. |
| `ARCHITECTURE.md` | This file. |

### `gp_emulator/`

| File | Role |
|------|------|
| `GP_emu.py` | **GP training script.** Loads data, fits 6 GPRs (one per k-bin), validates, saves `gp_models.joblib`. |
| `GP_MCMC.py` | **GP-based MCMC script.** Loads `gp_models.joblib`, runs emcee, saves chain to `pk{idx}.out`. Drop-in replacement for `MCMC.py`. |
| `gp_models.joblib` | Serialised model bundle: `{'scaler': StandardScaler, 'gps': [gp0…gp5]}`. Produced by `GP_emu.py`. |
| `run_corner.py` | Convenience script: runs MCMC and generates corner/trace plot in one go. |
| `make_plot.py` | Re-generates the corner plot from a saved `pk202.out` chain without re-running MCMC. |
| `corner_plot.png` | Most recent corner + trace plot output. |
| `pk202.out` | Saved MCMC chain for parameter index 202 (Mhmin = −0.3521). |
| `README.md` | Quick-start guide for the GP emulator sub-directory. |

### Generated artefacts (root)

| File | Produced by | Contents |
|------|-------------|----------|
| `cii_model.h5` | `CII_emu.py` | Trained Keras model (final epoch) |
| `cii_bestmodel/` | `CII_emu.py` | Best checkpoint saved by ModelCheckpoint |
| `cii_history.npy` | `CII_emu.py` | Training history dict (loss, RMSE per epoch) |
| `pk_test` | `CII_emu.py` | Held-out test-set power spectra (tab-separated) |
| `params_test` | `CII_emu.py` | Held-out test-set parameters (tab-separated) |
| `predictions` | `CII_emu.py` | ANN predictions on test set |
| `pk202.out` | `MCMC.py` | ANN-based MCMC chain for sample 202 |

---

## 8. Dependency graph

```
ann_input.py ──────────────────────────────────────────┐
                                                       ↓
data/               ─→  CII_emu.py  ─→  functions.py  ─→  cii_model.h5
(Npk.txt, k.txt,                                              ↓
 nbins.txt, params_t)                                     MCMC.py  ─→  pk202.out
      │
      └─→  Hyperparameter_tuning.py  ─→  tuning_results/
      │         (uses params_LH.txt)
      │
      └─→  gp_emulator/GP_emu.py  ─→  gp_models.joblib
                                           ↓
                                       GP_MCMC.py  ─→  pk202.out
                                           ↓
                                       make_plot.py  ─→  corner_plot.png
```

The ANN pipeline (top) and GP pipeline (bottom) are completely independent after
the data loading step. They read the same `data/` files and produce `pk202.out`
chains with the same format, so downstream analysis code works with both.

---

## 9. Data flow in detail

### Training (GP path — recommended)

```
data/Npk.txt   →  npk (2167×10)
data/k.txt     →  k   (6,)
data/nbins.txt →  nbins (6,)
data/params_t  →  params (2167×2)

npk[:, 2:8]    →  npk  (2167×6)   [select 6 scientific k-bins]
k**3 * npk / (2π²)  →  dpk (2167×6)  [dimensionless power spectrum]
log(dpk) / 10  →  pk   (2167×6)   [log-scaled target]
params[:, 0:1] →  X    (2167×1)   [Mhmin only; alpha excluded]

train_test_split(X, pk, test_size=0.1, shuffle=True, random_state=42)
  →  X_train (1950×1),  X_test (217×1)
  →  pk_train(1950×6),  pk_test(217×6)

StandardScaler.fit_transform(X_train)  →  X_scaled (1950×1)  [mean=0, std=1]

for i in range(6):
    GaussianProcessRegressor.fit(X_scaled, pk_train[:, i])
    → gps[i]    # fitted GP for k-bin i

joblib.dump({'scaler': scaler, 'gps': gps}, 'gp_models.joblib')
```

### Inference (GP path)

```
gp_models.joblib  →  scaler, gps

data/Npk.txt + params_t  →  fn_t (2167×6)   [same preprocessing as above]
fn_t[202]  →  data (6,)   ["observed" spectrum for sample 202]

build_cov_inv(data, nbins)  →  cov_inv (6×6 diagonal)

emcee.EnsembleSampler(n_walkers=4, n_dim=1, log_prob, vectorize=True)

log_posterior(theta):          # theta: (n_walkers, 1)
    X_scaled = scaler.transform(theta)
    preds = stack([gp.predict(X_scaled) for gp in gps])   # (n_walkers, 6)
    diff  = preds - data                                    # (n_walkers, 6)
    logL  = -0.5 * einsum('ij,jk,ik->i', diff, cov_inv, diff)
    return where(in_prior, logL, -inf)

sampler.run_mcmc(p0, n_burnin)  →  burn-in (discarded)
sampler.run_mcmc(state, n_samples)  →  chain (n_walkers × n_samples, 1)
np.savetxt('pk202.out', chain)
```

---

## 10. Key design decisions and trade-offs

### Why `log(dpk)/10` rather than a standard scaler?

A `StandardScaler` or `MinMaxScaler` would be fitted on the training set and
would shift/scale differently for each k-bin. Using `log(dpk)/10` is a fixed,
physics-motivated transform that all scripts agree on without any fitted object
— meaning the emulator output can be compared directly to the data without
needing the scaler to be passed around.

### Why `normalize_y=True` inside the GPR?

Even though all targets are in log(dpk)/10 space, the absolute values and
variances differ across k-bins (k-bin 0 has larger values and higher variance
than k-bin 5). `normalize_y=True` subtracts the per-bin mean and divides by the
per-bin std before fitting the GP kernel, making the kernel hyperparameters
interpretable and the optimisation well-conditioned.

### Why one GP per k-bin rather than a multi-output GP?

Multi-output GPs are significantly more expensive (kernel matrix scales as
(n × n_bins)²) and require modelling cross-bin covariances that are unnecessary
when each bin is independently inferred. Six independent 1950×1950 kernel
matrices are cheap; one 11700×11700 matrix is not.

### Why 4 walkers?

emcee requires `n_walkers ≥ 2 × n_dim`. With `n_dim = 1` the minimum is 2, but
4 walkers provide better mixing and allow diagnosing non-convergence by comparing
walker traces. The choice matches the original pipeline's `walker_ratio=2 × 2
params`.

### Why index 202?

`params_t[202]` has Mhmin = −0.3521, which sits well inside the prior range and
is well-sampled by the training data. It serves as the "observed" data point for
the MCMC demonstration. To infer parameters from a different spectrum, change
`idx` in `MCMC.py` or `GP_MCMC.py`.

### Why not shuffle in the ANN split (`CII_emu.py`)?

This is a **bug** inherited from the original code, not a design choice. The
`params_t` grid is sorted by Mhmin, so `shuffle=False` places all the
high-Mhmin samples into the test set. In that regime spectra are nearly
degenerate (std ≈ 10⁻¹⁵), so any predictor achieves near-zero error — the
reported accuracy is misleading. The GP scripts use `shuffle=True` to get an
honest evaluation.

---

## 11. Adding a new emulator

If you want to replace the GP with a different model (e.g. a neural process or
a random forest), follow this pattern:

1. Write a training script that produces a serialised model object with a
   `predict(X)` method where `X` has shape `(n, 1)` (Mhmin only) and the output
   has shape `(n,)` (log(dpk)/10 for one k-bin), or shape `(n, 6)` for all bins.
2. The model must be loadable at inference time with a single `joblib.load()` or
   equivalent.
3. Copy `gp_emulator/GP_MCMC.py` and replace the `make_log_posterior` body with
   your model's prediction call. Everything else (data loading, covariance,
   emcee setup, output) stays the same.
4. Validate on a random 10% split with `shuffle=True, random_state=42` to get
   a comparable accuracy number.

---

## 12. Quick-start cheatsheet

```bash
conda activate cii_emu
cd sign_separated_CII

# --- GP pipeline (recommended) ---

# Train
cd gp_emulator && python GP_emu.py
# → prints per-bin accuracy; saves gp_models.joblib

# MCMC + corner plot (10 000 steps)
python run_corner.py
# → saves pk202.out, corner_plot.png

# --- ANN pipeline (legacy) ---

# Optional: find best hyperparameters
python Hyperparameter_tuning.py
# → copy results into ann_input.py

# Train
python CII_emu.py
# → saves cii_model.h5

# MCMC
python MCMC.py   # enter 1000 when prompted
# → saves pk202.out
```
