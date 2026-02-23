#!/usr/bin/env python
"""
GP-based emulator for the CII power spectrum.

Trains one GaussianProcessRegressor per k-bin (6 total) on log(dpk)/10 vs
Mhmin. Alpha is constant at 0.395 across all training samples and is excluded
as a GP input dimension.

Saves the fitted (scaler, [gp0, ..., gp5]) bundle to gp_models.joblib.

Run from inside gp_emulator/:
    python GP_emu.py
"""

import warnings
warnings.filterwarnings('ignore')

import math
import time

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


# ---------------------------------------------------------------------------
# Data paths — relative to gp_emulator/ working directory
# ---------------------------------------------------------------------------

DATA_DIR    = '../data/'
NPK_PATH    = DATA_DIR + 'Npk.txt'
K_PATH      = DATA_DIR + 'k.txt'
NBINS_PATH  = DATA_DIR + 'nbins.txt'
PARAMS_PATH = DATA_DIR + 'params_t'

MODEL_OUT   = 'gp_models.joblib'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and preprocess CII power spectrum data.

    Returns
    -------
    pk     : ndarray (n_samples, 6)   log(dpk)/10, one column per k-bin
    params : ndarray (n_samples, 1)   Mhmin column only (alpha is constant)
    k      : ndarray (6,)             k-bin centres
    nbins  : ndarray (6,)             number of k-modes per bin
    """
    npk    = np.loadtxt(NPK_PATH, usecols=(2, 3, 4, 5, 6, 7))
    k      = np.loadtxt(K_PATH)
    nbins  = np.loadtxt(NBINS_PATH)
    params = np.loadtxt(PARAMS_PATH)

    # Dimensionless power spectrum: Delta^2(k) = k^3 P(k) / (2 pi^2)
    dpk = k**3 * npk / (2 * math.pi**2)
    pk  = np.log(dpk) / 10   # log-scaled; matches MCMC.py target space

    # Only Mhmin (column 0); alpha (column 1) is constant — zero variance
    mhmin = params[:, 0:1]   # shape (n, 1)

    return pk, mhmin, k, nbins


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gps(X_train, pk_train):
    """Train one GPR per k-bin.

    Parameters
    ----------
    X_train  : ndarray (n, 1)   raw Mhmin values
    pk_train : ndarray (n, 6)   log(dpk)/10 targets

    Returns
    -------
    gps    : list[GaussianProcessRegressor]  one fitted GP per k-bin
    scaler : StandardScaler fitted on X_train
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Matern nu=2.5 is once mean-square differentiable — suitable for smooth
    # physical functions. WhiteKernel absorbs numerical noise in the sims.
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
        Matern(length_scale=0.1, length_scale_bounds=(1e-3, 1e1), nu=2.5) +
        WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

    gps = []
    n_bins = pk_train.shape[1]

    for i in range(n_bins):
        print(f"  Fitting GP for k-bin {i} ...", end=' ', flush=True)
        t0 = time.time()
        gp = GaussianProcessRegressor(
            kernel=kernel,          # fit() clones this internally
            n_restarts_optimizer=3,
            alpha=1e-6,
            normalize_y=True,
        )
        gp.fit(X_scaled, pk_train[:, i])
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f} s)")
        gps.append(gp)

    return gps, scaler


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(gps, scaler, X_test, pk_test):
    """Report emulator accuracy on the held-out test set.

    Converts predictions from log(dpk)/10 back to physical dpk for error
    reporting so that the threshold (< 10% mean abs % error) is in physical
    space, consistent with the plan specification.

    Returns
    -------
    all_pass : bool   True if all k-bins clear the 10% mean-error threshold
    """
    X_scaled = scaler.transform(X_test)
    n_bins   = pk_test.shape[1]

    print("\n--- Validation results ---")
    header = f"{'k-bin':>6}  {'mean |%err|':>12}  {'max |%err|':>11}  "
    header += f"{'p90 |%err|':>10}  {'status':>6}"
    print(header)
    print("-" * 55)

    all_pass = True
    for i in range(n_bins):
        pred_log  = gps[i].predict(X_scaled)
        pred_phys = np.exp(10 * pred_log)
        true_phys = np.exp(10 * pk_test[:, i])

        abs_pct = np.abs((pred_phys - true_phys) / true_phys) * 100
        mean_e  = abs_pct.mean()
        max_e   = abs_pct.max()
        p90_e   = np.percentile(abs_pct, 90)

        status = "PASS" if mean_e < 10.0 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {i:>4}  {mean_e:>12.3f}  {max_e:>11.3f}  {p90_e:>10.3f}  {status:>6}")

    print("-" * 55)
    verdict = ("PASS — all bins < 10% mean abs % error"
               if all_pass
               else "FAIL — some bins exceed 10% threshold")
    print(f"Overall: {verdict}")

    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    pk, params, k, nbins = load_data()
    print(f"  {len(pk)} samples, {pk.shape[1]} k-bins")
    print(f"  Mhmin range: [{params.min():.4f}, {params.max():.4f}]")

    # Random split — fixes the broken shuffle=False in CII_emu.py
    X_train, X_test, pk_train, pk_test = train_test_split(
        params, pk, test_size=0.1, random_state=42, shuffle=True
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\nTraining GPs ...")
    t0 = time.time()
    gps, scaler = train_gps(X_train, pk_train)
    print(f"Total training time: {time.time() - t0:.1f} s")

    print(f"\nSaving model → {MODEL_OUT} ...")
    joblib.dump({'scaler': scaler, 'gps': gps}, MODEL_OUT)

    validate(gps, scaler, X_test, pk_test)


if __name__ == '__main__':
    main()
