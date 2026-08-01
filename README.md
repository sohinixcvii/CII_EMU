# CII_EMU

Emulator for the CII/21cm cross-power spectrum. Given astrophysical parameters
(primarily `Mhmin`, the minimum halo mass), it predicts the dimensionless power
spectrum Δ²(k) across six k-bins and supports Bayesian parameter inference via
MCMC.

Two emulator backends are provided; the **GP emulator is recommended**.

---

## Repository layout

```
CII_EMU/
├── data/                        # Training data (tracked; read-only inputs)
│   ├── Npk.txt                  # Raw power spectra (2167 samples × 10 cols)
│   ├── k.txt                    # k-bin centres (6 values, Mpc⁻¹)
│   ├── nbins.txt                # Number of Fourier modes per k-bin
│   ├── params_t                 # Parameter grid (Mhmin, alpha)
│   ├── params_LH.txt            # Latin Hypercube samples (HPO only)
│   ├── LH_hyperparameters       # Metadata about the LH sampling
│   └── visualize.py             # Data visualisation utility
│
└── sign_separated_CII/          # Current pipeline (ANN + GP)
    ├── ann_input.py             # ANN configuration (edit this to tune)
    ├── CII_emu.py               # ANN training script
    ├── functions.py             # ANN helper functions
    ├── Hyperparameter_tuning.py # Keras-tuner HPO (optional)
    ├── MCMC.py                  # ANN-based MCMC inference (emcee)
    ├── ARCHITECTURE.md          # Technical reference
    ├── MANUAL.md                # Full user guide  ← start here
    ├── environment.yaml         # Conda environment spec
    └── gp_emulator/             # GP emulator (recommended)
        ├── GP_emu.py            # GP training script
        ├── GP_MCMC.py           # GP-based MCMC inference
        ├── run_corner.py        # Convenience: train + MCMC + corner plot
        ├── make_plot.py         # Re-generate corner plot from saved chain
        └── README.md            # GP quick-start guide
```

---

## Quick start

```bash
conda env create -f sign_separated_CII/environment.yaml
conda activate cii_emu

# Create a local data symlink (scripts expect data/ relative to their directory)
ln -s ../data sign_separated_CII/data

cd sign_separated_CII/gp_emulator

# Train the GP emulator (~2 min)
python GP_emu.py

# Run MCMC inference
echo "1000" | python GP_MCMC.py
```

Full instructions, including the ANN pipeline and troubleshooting, are in
[`sign_separated_CII/MANUAL.md`](sign_separated_CII/MANUAL.md).

---

## Why GP over ANN?

The effective parameter space is 1-dimensional (`alpha` is constant at 0.395
across all training samples). A Gaussian Process with a Matérn-2.5 kernel
interpolates this smooth 1D function near-exactly, achieving < 0.25% mean
absolute error per k-bin. The ANN achieves 44–53% error on a proper random
test split due to exponential amplification of log-space errors.

See [`sign_separated_CII/ARCHITECTURE.md`](sign_separated_CII/ARCHITECTURE.md)
for full technical details.

---

## Environment

The conda environment `cii_emu` (Python 3.10, TensorFlow 2.13, scikit-learn,
emcee ≥ 3.1) is defined in `sign_separated_CII/environment.yaml`.

```bash
conda env create -f sign_separated_CII/environment.yaml
conda activate cii_emu
```
