# WRF-ML Wind Speed Bias Correction

A pipeline for simulating severe convective wind events using the Weather Research and
Forecasting (WRF) model and correcting surface wind speed predictions with machine
learning, validated against ASOS station observations and NEXRAD radar data. Developed
in support of downed power line prediction for a Midwest utility partner.

---

## Overview

Operational NWP models systematically underpredict surface wind speeds during derecho
events. This project addresses that gap by:

1. Running high-resolution WRF simulations (3 km / 1 km nested domains) for historical
   derecho events using HRRR forcing
2. Downloading and processing ASOS surface observations and NEXRAD radar data for
   validation
3. Extracting WRF surface wind output collocated with ASOS stations
4. Diagnosing storm structure, cold pool dynamics, and vertical wind profiles
5. Training ML models to predict and correct the WRF–observed wind bias (in progress)
6. Evaluating spatial bias patterns to inform power line outage risk (in progress)

---

## Events

| Event       | Date         | Forcing  | Status             |
|-------------|--------------|----------|--------------------|
| Houston, TX | May 2024     | HRRR     | Complete           |
| Iowa        | August 2020  | HRRR     | In progress        |
| Colorado    | December 2020| HRRR     | In progress        |
| Midwest     | June 2012    | HRRR     | In progress        |

---

## Repository Structure

```
WRF-ML-windspeeds/
├── analysis/
│   ├── data/asos/                # ASOS observation data per event
│   │   ├── {event}_asos_summary.csv      # per-station peak gust statistics
│   │   ├── {event}_asos_timeseries.csv   # full 5-min time series (Houston only)
│   │   └── {event}_metadata.json         # station metadata
│   ├── figures/                  # Analysis figure outputs
│   ├── asos_download.py          # ASOS download from IEM
│   ├── radar.py                  # WRF vs. NEXRAD reflectivity comparison
│   ├── windswath.py              # WRF wind swath vs. ASOS peak gust comparison
│   ├── crosssection.py           # Vertical cross sections of T anomaly, |U|, W
│   ├── 3D_plots.py               # 3D cold pool visualizations (Plotly)
│   └── ML_analysis.py            # ML bias correction pipeline (LOGO-CV by station)
├── wrf/
│   └── events/                   # WRF namelist.input and namelist.wps per event
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Analysis Scripts

### `asos_download.py`
Downloads 5-minute ASOS observations from the Iowa Environmental Mesonet for a given
event and time window. Computes peak wind statistics per station and saves a summary
CSV compatible with the ML bias correction pipeline. For Houston, additionally retrieves
the full 5 minute time series for cold pool investigation.

```bash
python asos_download.py --event houston \
    --start "2024-05-16 00:00" --end "2024-05-17 06:00"
```

**Output:**
- `data/asos/{event}_asos_summary.csv` — per-station peak stats
- `data/asos/{event}_asos_timeseries.csv` — full time series (when applicable)
- `data/asos/{event}_metadata.json` — station metadata

---

### `radar.py`
Compares WRF simulated composite reflectivity against IEM NEXRAD composite reflectivity
for qualitative storm track and structure validation. Produces side-by-side static
comparisons and animations using an NWS-standard reflectivity colormap.

```bash
python radar.py \
    --wrfout /path/to/wrfout_d01_* \
    --output-dir figures/radar \
    --event-window-start "2024-05-16 18:00" \
    --event-window-end "2024-05-17 02:00" \
    --peak-time "2024-05-16 23:00"
```

**Output:**
- `refl_comparison_{time}.png` — side-by-side WRF vs. observed at peak time
- `wrf_refl_animation.gif`, `obs_refl_animation.gif`, `refl_comparison.gif`

---

### `windswath.py`
Computes the WRF maximum 10-m wind speed at each grid point over the event window
("windswath") and compares against ASOS observed peak gusts. Produces a map with 
station overlays plus an embedded comparison table showing the WRF wind speed
at each station's grid point and within a 10-km neighborhood radius. 

```bash
python windswath.py \
    --wrfout /path/to/wrfout_d02_* \
    --asos data/asos/houston_asos_summary.csv \
    --output figures/windswath/wind_swath_comparison_d02.png \
    --event-start "2024-05-16 18:00" \
    --event-end "2024-05-17 02:00"
```

**Output:** Wind swath comparison figure with embedded ASOS station table

---

### `crosssection.py`
Extracts north–south vertical cross sections through the WRF output to diagnose the
vertical structure of the cold pool, the horizontal wind profile, and the vertical
velocity field. 

```bash
python crosssection.py \
    --wrfout /path/to/wrfout_d02_* \
    --longitude -95.35 \
    --time "2024-05-17 00:00" \
    --output figures/coldpool/crosssection_houston.png
```

**Output:** Three-panel cross section (T anomaly, |U|, W) with cold pool top contoured

---

### `3D_plots.py`
Generates interactive 3D visualizations of the WRF cold pool temperature anomaly field
and accompanying wind structure using Plotly. Used for diagnostic exploration of cold
pool depth, intensity, and centroid location relative to verification stations.

```bash
python 3D_plots.py \
    --wrfout /path/to/wrfout_d02_* \
    --time "2024-05-17 00:00" \
    --output figures/coldpool/coldpool_3d.html
```

**Output:** Interactive HTML visualization

---

### `ML_analysis.py`
Main ML bias correction pipeline. Collocates WRF output with ASOS stations, trains
models to predict the WRF–observed wind bias, and evaluates performance using
Leave-One-Group-Out Cross-Validation by station (LOGO-CV) to prevent data leakage.

```bash
# Single event
python ML_analysis.py --event houston --wrfout /path/to/wrfout

# Multi-event training
python ML_analysis.py --multi-event --events houston iowa colorado
```

**Key design choices:**
- Target variable is WRF–observed bias, not raw observed wind speed
- Features include WRF wind components (u, v), surface pressure, temperature, and
  spatial coordinates (lat, lon, distance to coast)
- Models: Random Forest, Gradient Boosting, XGBoost
- LOGO-CV prevents a single station's observations from appearing in both training
  and test sets

---

## WRF Configuration

- **Model version:** WRF v4.7.1
- **Projection:** Lambert Conformal
- **Domains:** 3 km outer (d01) / 1 km inner (d02) nested
- **Vertical levels:** 47 terrain-following levels, lowest level ~60–73 m AGL
- **Forcing:** HRRR initial and lateral boundary conditions, hourly updates
- **Physics:**
  - Microphysics: NSSL 2-moment (mp_physics = 17)
  - Radiation: Goddard shortwave and longwave (ra_sw_physics = 5, ra_lw_physics = 5)
  - Surface layer: Revised MM5 (sf_sfclay_physics = 1)
  - Land surface model: RUC (sf_surface_physics = 3)
  - PBL: Yonsei University (YSU)
  - Cumulus: None (convection-permitting on both domains)
- **Output frequency:** d01 hourly, d02 every 15 min
- **HPC:** UIUC Keeling cluster

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
plotly
kaleido
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Limitations

- This repository is a work in progress - Future work includes the ml_analysis.py use
  with several different events that have not been simulated yet with WRF
- Radar comparison (`radar.py`) currently configured for Houston only; extension to
  remaining events is in progress

---

## Author

Dara Procell
University of Illinois Urbana-Champaign
Department of Atmospheric Sciences
