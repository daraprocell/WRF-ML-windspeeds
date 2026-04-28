# WRF-ML-windspeeds

# WRF-ML Wind Speed Bias Correction

A pipeline for simulating severe convective wind events using the Weather Research and
Forecasting (WRF) model and correcting surface wind speed predictions with machine
learning, validated against ASOS station observations and NEXRAD radar data. Developed in 
support of downed power line prediction for a Midwest utility partner.

---

## Overview

Operational NWP models systematically underpredict surface wind speeds during derecho
events. This project addresses that gap by:

1. Running high-resolution WRF simulations (3 km / 1 km nested domains) for historical derecho events using HRRR data
2. Downloading and processing ASOS surface observations and NEXRAD data for validation
3. Extracting WRF surface wind output collocated with ASOS stations
4. Analyzing cold pool dynamics from both WRF output and observations
5. Training ML models to predict and correct the WRF–observed wind bias (in progress)
6. Evaluating spatial bias patterns to inform power line outage risk (in progress)

---

## Events

| Event       | Date         | Data     | Status             |
|-------------|--------------|----------|--------------------|
| Houston, TX | May 2024     | GFS/HRRR | Complete           |
| Iowa        | August 2020  | HRRR     | In progress        |
| Colorado    | Dec 2020     | HRRR     | In progress        |
| Midwest     | June 2012    | HRRR     | In progress        |

---

## Repository Structure

```
WRF-ML-windspeeds/
├── namelist/                     # WRF namelist.input and namelist.wps files per event
├── scripts/
│   ├── preprocessing/            # WPS/metgrid setup, Vtable configuration
│   └── analysis/
│       ├── WRF_ASOS_analysis.py      # ML bias correction pipeline
│       ├── ASOS_download.py          # ASOS peak wind summary download
│       ├── coldpool_asos_download.py # ASOS full time series + cold pool diagnostics
│       ├── coldpool_analysis.py      # WRF cold pool visualization and ASOS comparison
│       └── wrf_radar_plotting.py     # WRF vs. NEXRAD reflectivity plots and animations
├── data/
│   └── asos/                     # ASOS station observations (not tracked in git)
├── figures/
│   ├── radar/                    # Reflectivity comparison outputs
│   └── coldpool/                 # Cold pool analysis outputs
└── README.md
```

---

## Analysis Scripts

### `ASOS_download.py`
Downloads 5-minute ASOS observations from the Iowa Environmental Mesonet for a given
event and time window, computes peak wind statistics per station, and saves a summary
CSV compatible with the ML bias correction pipeline.

```bash
python ASOS_download.py --event houston --start "2024-05-16 06:00" --end "2024-05-17 06:00"
```

**Output:** `data/asos/{event}_asos_summary.csv`, `data/asos/{event}_metadata.json`

---

### `coldpool_asos_download.py`
Extended ASOS download script that retrieves the full 5-minute time series (temperature,
dewpoint, wind, pressure, weather codes) and computes derived cold pool diagnostics:
temperature anomalies, pressure surges, rolling peak gusts, and cold pool passage detection.

```bash
python coldpool_asos_download.py --event houston \
    --start "2024-05-16 00:00" --end "2024-05-17 06:00"
```

**Output:**
- `data/asos/{event}_asos_timeseries.csv` — full time series, all stations
- `data/asos/{event}_asos_summary.csv` — per-station peak stats (ML-pipeline compatible)

---

### `coldpool_analysis.py`
Visualizes WRF cold pool dynamics and compares simulated fields against ASOS ground
truth. Reads WRF output and the ASOS time series produced by `coldpool_asos_download.py`.

```bash
python coldpool_analysis.py \
    --event houston \
    --asos data/asos/houston_asos_timeseries.csv \
    --wrfout /path/to/wrfout_d01_* \
    --output-dir figures/coldpool \
    --peak-time "2024-05-16 23:00" \
    --event-window-start "2024-05-16 20:00" \
    --event-window-end "2024-05-17 02:00"
```

**Output:**
- `coldpool_timeseries_{event}.png` — per-station T / wind / pressure time series panels
- `coldpool_spatial_{event}.png` — spatial snapshot at peak event time
- `coldpool_sweep_{event}.gif` — animated cold pool propagation
- `coldpool_crosssection_{event}.png` — N–S vertical cross section at peak time

---

### `wrf_radar_plotting.py`
Compares WRF simulated reflectivity (`REFL_10CM`) against IEM NEXRAD composite
reflectivity for qualitative storm track and structure validation. Uses an NWS-standard
reflectivity colormap.

```bash
python wrf_radar_plotting.py \
    --wrfout /path/to/wrfout_d01_* \
    --output-dir figures/radar \
    --event-window-start "2024-05-16 18:00" \
    --event-window-end "2024-05-17 02:00" \
    --peak-time "2024-05-16 23:00"
```

**Output:**
- `wrf_refl_vs_obs_{time}.png` — side-by-side WRF vs. observed at peak time
- `wrf_refl_animation.gif` — WRF reflectivity evolution
- `obs_refl_animation.gif` — observed reflectivity over same window
- `refl_comparison.gif` — side-by-side animated comparison

---

### `WRF_ASOS_analysis.py`
Main ML bias correction pipeline. Collocates WRF output with ASOS stations, trains
models to predict the WRF–observed wind bias, and evaluates performance using
Leave-One-Out Cross-Validation by station (LOGO-CV) to prevent data leakage.

```bash
# Single event
python WRF_ASOS_analysis.py --event houston --wrfout /path/to/wrfout

# Multi-event training
python WRF_ASOS_analysis.py --multi-event --events houston iowa colorado
```

**Key design choices:**
- Target variable is WRF–observed bias, not raw observed wind speed
- Features include WRF wind components (u, v), surface pressure, temperature, and
  spatial coordinates (lat, lon, distance to coast)
- Models: Random Forest, Gradient Boosting, XGBoost
- LOGO-CV prevents a single event's station observations from appearing in both
  training and test sets

---

## WRF Configuration

- **Projection:** Lambert Conformal
- **Domains:** 3 km outer / 1 km inner nested domains
- **Forcing:** GFS (historical runs); transitioning to HRRR for improved mesoscale representation
- **Physics:**
  - Microphysics: *[fill in]*
  - PBL scheme: *[fill in]*
  - Cumulus scheme: *[fill in]*
- **HPC:** UIUC Keeling cluster

### Known Biases (Houston GFS Run)

- Domain-mean wind speed bias: approximately −23 mph (systematic underprediction)
- Storm track displacement: ~3° northward
- Timing lag: 4–6 hours relative to observations
- Motivated transition to HRRR forcing for improved initial conditions

---

## Requirements

```
Python >= 3.9
netCDF4
numpy
pandas
scikit-learn
xgboost
matplotlib
cartopy
wrf-python
requests
Pillow
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Limitations

- ML models trained on a single event with 13 stations are prone to overfitting;
  multi-event LOGO-CV is the intended production approach
- HRRR forcing migration is ongoing for all events beyond Houston
- Radar comparison (`wrf_radar_plotting.py`) currently configured for Houston only

---


## Author

Dara Procell  
University of Illinois Urbana-Champaign  
Department of Atmospheric Sciences
