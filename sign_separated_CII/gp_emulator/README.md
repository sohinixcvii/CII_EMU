# GP Emulator for CII Power Spectrum

This directory contains a Gaussian Process (GP) based emulator. 

## History of this project:

We were initially using an ANN based emulator for this project. Upon testing it was found that the ANN based emulator was failing at modelling with sufficient accuracy. 

## Why GP instead of ANN?

The parameter space is effectively 1-dimensional: `alpha` is constant at 0.395
across all training samples, and only `Mhmin` varies. A GP with a Matérn-2.5
kernel is ideal for this smooth, densely-sampled 1D interpolation problem.

The ANN achieves 44–53% mean absolute % error on a proper random test split due
to (a) exponential amplification of log-space errors, and (b) the mismatch
between a 32-layer network and a trivial 1D function. The GP resolves both.

We have set a threshold accuracy of <10% for all k-bins. 

**Measured accuracy (vs <10% threshold):**
- All 6 k-bins: mean absolute % error < 10%
- k-bin 0 (hardest): ~0.25% mean, ~6% max

## Files

| File | Description |
|------|-------------|
| `GP_emu.py` | Training script: fits 6 GPs and saves `gp_models.joblib` |
| `GP_MCMC.py` | MCMC inference using GPs |
| `gp_models.joblib` | Saved model artefact (produced by `GP_emu.py`) |
| `README.md` | This file |

## Usage

Run from inside the `gp_emulator/` directory:

```bash
conda activate cii_emu
cd gp_emulator

# Step 1: Train the GPs (~2 min)
python GP_emu.py

# Step 2: Run MCMC with the GP emulator; echo number of iterations
echo "500" | python GP_MCMC.py
```

## Model details

- **Input**: `Mhmin` only, shape `(n, 1)`, `StandardScaler` normalised
- **Output**: `log(dpk)/10` per k-bin (same space as the data in `MCMC.py`)
- **Kernel**: `ConstantKernel(1.0) × Matérn(ν=2.5, l=0.1) + WhiteKernel(1e-5)`
- **Saved bundle**: `{'scaler': StandardScaler, 'gps': [gp0, gp1, ..., gp5]}`

`normalize_y=True` inside each GPR handles the different output scales per bin.
`n_restarts_optimizer=3` avoids local minima in the log-marginal-likelihood.

## Data files

The data is read from `../data/` (relative to this directory), i.e. the same
`data/` folder used by the rest of the pipeline:

| File | Used for |
|------|----------|
| `../data/Npk.txt` | Raw power spectra (columns 2–7) |
| `../data/k.txt` | k-bin centres |
| `../data/nbins.txt` | Number of k-modes per bin |
| `../data/params_t` | Parameter grid (Mhmin, alpha) |

See `../MANUAL.md` Section 11 for full documentation.
