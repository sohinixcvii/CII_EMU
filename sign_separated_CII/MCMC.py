#!/usr/bin/env python
"""
MCMC parameter inference for CII power spectrum.

Uses emcee (affine-invariant ensemble sampler) with vectorized likelihood
evaluation: all walkers are batched into a single model.predict() call per step,
which is substantially faster than the old one-at-a-time CosmoHammer approach.

Free parameter : Mhmin  (uniform prior over MHMIN_BOUNDS)
Fixed parameter: alpha = 0.395
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import emcee
from tensorflow import keras as ks

from ann_input import npk_p, k_p, n_p, path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MHMIN_BOUNDS = (-0.4328129267019357, 0.4328129267019357)
ALPHA_FIXED  = 0.395   # alpha is held fixed; only Mhmin is sampled
MODEL_PATH   = 'cii_model.h5'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load and preprocess CII power spectrum data.

    Returns
    -------
    fn_t        : ndarray (n_samples, n_k)  log-scaled dimensionless power spectra
    nbins       : ndarray (n_k,)            number of k-modes per bin
    params_test : ndarray (n_samples, 2)    parameter grid
    """
    npk    = np.loadtxt(npk_p, usecols=(2, 3, 4, 5, 6, 7))
    k      = np.loadtxt(k_p)
    nbins  = np.loadtxt(n_p)
    params = np.loadtxt(path + 'params_t')

    # Dimensionless power spectrum: Delta^2(k) = k^3 P(k) / (2 pi^2)
    dpk  = k**3 * npk / (2 * np.pi**2)
    fn_t = np.log(dpk) / 10   # log-scaled, matches emulator training target

    return fn_t, nbins, params


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------

def build_cov_inv(data, nbins):
    """Build the inverse of the diagonal Gaussian covariance matrix.

    cov_ii = |data_i^2| / nbins_i  +  |data_i / sqrt(nbins_i)|
    """
    cov_diag = np.abs(data**2) / nbins + np.abs(data / np.sqrt(nbins))
    return np.diag(1.0 / cov_diag)


# ---------------------------------------------------------------------------
# Vectorized log-posterior
# ---------------------------------------------------------------------------

def make_log_posterior(model, data, cov_inv):
    """Return a vectorized log-posterior function compatible with emcee.

    The returned function accepts theta of shape (n_walkers, 1) and returns
    an array of shape (n_walkers,), allowing emcee to batch all walkers into
    a single model.predict() call per step.
    """
    lo, hi = MHMIN_BOUNDS

    def log_posterior(theta):
        # theta: (n_walkers, 1)
        mhmin    = theta[:, 0]
        in_prior = (mhmin > lo) & (mhmin < hi)

        # Batch-predict for all walkers simultaneously
        params   = np.column_stack([mhmin, np.full(len(mhmin), ALPHA_FIXED)])
        model_th = model.predict(params, verbose=0)      # (n_walkers, n_k), log-scaled

        # Gaussian log-likelihood: -0.5 * diff^T C^{-1} diff
        # Both model_th and data are in log(dpk)/10 space
        diff     = model_th - data[np.newaxis, :]        # (n_walkers, n_k)
        loglikes = -0.5 * np.einsum('ij,jk,ik->i', diff, cov_inv, diff)

        return np.where(in_prior, loglikes, -np.inf)

    return log_posterior


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def run_mcmc(data, nbins, model_path=MODEL_PATH, index=202,
             n_walkers=4, burnin_frac=0.1, n_samples=1000):
    """Run emcee MCMC to infer Mhmin from CII power spectrum data.

    Parameters
    ----------
    data        : 1D array  log-scaled dimensionless power spectrum for one sample
    nbins       : 1D array  number of k-modes per bin (for covariance)
    model_path  : str       path to the Keras emulator (.h5)
    index       : int       sample index, used only for the output filename
    n_walkers   : int       number of emcee walkers (must be even, >= 2)
    burnin_frac : float     burn-in length as a fraction of n_samples
    n_samples   : int       number of production samples per walker

    Returns
    -------
    chain : ndarray (n_walkers * n_samples, 1)  flattened posterior samples
    """
    model    = ks.models.load_model(model_path)
    cov_inv  = build_cov_inv(data, nbins)
    log_prob = make_log_posterior(model, data, cov_inv)

    n_dim    = 1
    n_burnin = max(1, int(burnin_frac * n_samples))
    lo, hi   = MHMIN_BOUNDS

    # Initialise walkers uniformly within the prior
    p0 = np.random.uniform(lo, hi, size=(n_walkers, n_dim))

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, vectorize=True)

    print(f"Burn-in: {n_burnin} steps × {n_walkers} walkers ...")
    t0    = time.time()
    state = sampler.run_mcmc(p0, n_burnin, progress=True)
    sampler.reset()

    print(f"Production: {n_samples} steps × {n_walkers} walkers ...")
    sampler.run_mcmc(state, n_samples, progress=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f} s | mean acceptance fraction: "
          f"{np.mean(sampler.acceptance_fraction):.3f}")

    chain    = sampler.get_chain(flat=True)   # (n_walkers * n_samples, 1)
    out_file = f'pk{index}.out'
    np.savetxt(out_file, chain)
    print(f"Chain saved → {out_file}")

    return chain


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    fn_t, nbins, params_test = load_data()

    idx = 202
    print(f"Params[{idx}]: {params_test[idx]}")

    n_samples = int(input("Enter number of samples: "))

    # n_walkers=4 matches the original walker_ratio=2 with 2 params
    run_mcmc(
        data=fn_t[idx],
        nbins=nbins,
        index=idx,
        n_walkers=4,
        n_samples=n_samples,
    )
