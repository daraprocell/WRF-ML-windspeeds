#!/usr/bin/env python3
"""
coldpool_analysis.py
--------------------
Visualize WRF cold pool dynamics and compare against ASOS ground truth.

Produces:
  1. coldpool_timeseries_{event}.png  — per-station time series panels (T, wind, pressure)
  2. coldpool_spatial_{event}.png     — spatial snapshot at peak event time
  3. coldpool_sweep_{event}.gif       — animated cold pool propagation
  4. coldpool_crosssection_{event}.png — N-S vertical cross section at peak time

Example use:
    python coldpool_analysis.py \
        --event houston \
        --asos data/asos/houston_asos_timeseries.csv \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d01_* \
        --output-dir figures/coldpool \
        --peak-time "2024-05-16 23:00" \
        --event-window-start "2024-05-16 20:00" \
        --event-window-end "2024-05-17 02:00"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Keeling
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

# Color scheme
C_WRF  = '#E05C2A'   # WRF — warm coral
C_OBS  = '#2A6EBB'   # observations — blue
C_CP   = '#5BB85A'   # cold pool arrival — green
C_GRID = '#DDDDDD'

# ---------------------------------------------------------------------------
# Utility: haversine distance
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# WRF extraction
# ---------------------------------------------------------------------------

def load_wrf_times(wrfout_files):
    """Return list of (file, time_index, datetime) tuples."""
    entries = []
    for fpath in sorted(wrfout_files):
        with Dataset(fpath, 'r') as nc:
            times_raw = nc.variables['Times'][:]
            for t in range(times_raw.shape[0]):
                tstr = ''.join([c.decode() for c in times_raw[t]]).replace('_', ' ')
                dt = pd.to_datetime(tstr)
                entries.append((fpath, t, dt))
    return entries


def extract_wrf_surface(wrfout_files, event_window_start, event_window_end):
    """
    Extract 2D surface fields (T2, U10, V10, PSFC) for all time steps
    within the event window. Returns dict of arrays.
    """
    wrf_times = load_wrf_times(wrfout_files)
    wrf_times = [(f, t, dt) for f, t, dt in wrf_times
                 if event_window_start <= dt <= event_window_end]

    if not wrf_times:
        raise ValueError("No WRF time steps found in event window!")

    print(f"Loading {len(wrf_times)} WRF time steps...")

    # Load grid from first file
    with Dataset(wrf_times[0][0], 'r') as nc:
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]

    T2_all    = []
    U10_all   = []
    V10_all   = []
    PSFC_all  = []
    WSPD_all  = []
    dt_all    = []

    for fpath, tidx, dt in wrf_times:
        with Dataset(fpath, 'r') as nc:
            T2   = nc.variables['T2'][tidx, :, :]    # K
            U10  = nc.variables['U10'][tidx, :, :]   # m/s
            V10  = nc.variables['V10'][tidx, :, :]   # m/s
            PSFC = nc.variables['PSFC'][tidx, :, :]  # Pa → mb
            WSPD = np.sqrt(U10**2 + V10**2)

        T2_all.append(T2)
        U10_all.append(U10)
        V10_all.append(V10)
        PSFC_all.append(PSFC / 100.0)  # Pa → mb
        WSPD_all.append(WSPD)
        dt_all.append(dt)

    return {
        'lat':   wrf_lat,
        'lon':   wrf_lon,
        'times': dt_all,
        'T2':    np.array(T2_all),      # K
        'U10':   np.array(U10_all),
        'V10':   np.array(V10_all),
        'PSFC':  np.array(PSFC_all),    # mb
        'WSPD':  np.array(WSPD_all),    # m/s
    }


def extract_wrf_column(wrfout_files, lat, lon, event_window_start, event_window_end):
    """
    Extract vertical profile at a single lat/lon for cross-section.
    Returns dict with time, levels, T_pert, U, W arrays.
    """
    wrf_times = load_wrf_times(wrfout_files)
    wrf_times = [(f, t, dt) for f, t, dt in wrf_times
                 if event_window_start <= dt <= event_window_end]

    with Dataset(wrf_times[0][0], 'r') as nc:
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]
        nz = nc.variables['T'].shape[1]

    # Find nearest grid row for cross section (fix lon)
    target_lon = lon
    lon_diffs  = np.abs(wrf_lon[wrf_lat.shape[0]//2, :] - target_lon)
    col_idx    = np.argmin(lon_diffs)

    lats_col = wrf_lat[:, col_idx]

    T_pert_all = []
    U_all      = []
    W_all      = []
    HGT_all    = []

    for fpath, tidx, dt in wrf_times:
        with Dataset(fpath, 'r') as nc:
            T    = nc.variables['T'][tidx, :, :, col_idx]      # perturbation potential temp (K)
            U    = nc.variables['U10'][tidx, :, col_idx]       # just use U10 for surface
            PH   = nc.variables['PH'][tidx, :, :, col_idx]    # geopotential perturbation
            PHB  = nc.variables['PHB'][tidx, :, :, col_idx]   # base geopotential
            W    = nc.variables['W'][tidx, :, :, col_idx]     # vertical velocity

        HGT = (PH + PHB) / 9.81  # geopotential height in meters (staggered)
        HGT_mid = 0.5 * (HGT[:-1, :] + HGT[1:, :])  # unstagger

        T_pert_all.append(T)
        W_all.append(W[:-1, :])  # unstagger W
        HGT_all.append(HGT_mid)

    return {
        'lats':   lats_col,
        'times':  [dt for _, _, dt in wrf_times],
        'T_pert': np.array(T_pert_all),
        'W':      np.array(W_all),
        'HGT':    np.array(HGT_all),
    }


def extract_wrf_at_station(wrf_data, slat, slon):
    """
    Extract time series of WRF surface variables at a single station location
    using nearest grid point.
    """
    dist = haversine(slat, slon, wrf_data['lat'], wrf_data['lon'])
    idx  = np.unravel_index(np.argmin(dist), dist.shape)
    i, j = idx

    return {
        'times': wrf_data['times'],
        'T2':    wrf_data['T2'][:, i, j] - 273.15,   # K → °C
        'WSPD':  wrf_data['WSPD'][:, i, j],
        'PSFC':  wrf_data['PSFC'][:, i, j],
    }


# ---------------------------------------------------------------------------
# Cold pool detection (windowed)
# ---------------------------------------------------------------------------

def detect_cold_pool(ts_df, station, event_window_start, event_window_end):
    """
    Detect cold pool passage within the event window only.
    Returns arrival time and temperature drop, or NaT/NaN if not detected.
    """
    df = ts_df[ts_df['station'] == station].copy()
    df = df[(df['valid'] >= event_window_start) & (df['valid'] <= event_window_end)]
    if df.empty:
        return pd.NaT, np.nan

    temp_change = df['temp_c'].diff(6)  # change over 30 min
    cp_mask     = temp_change <= -3.0
    cp_rows     = df[cp_mask]

    if len(cp_rows) == 0:
        return pd.NaT, np.nan

    arrival    = cp_rows['valid'].min()
    temp_drop  = temp_change[cp_mask].min()
    return arrival, temp_drop


# ---------------------------------------------------------------------------
# Figure 1: Time series panels
# ---------------------------------------------------------------------------

def plot_timeseries(asos_df, wrf_data, stations_to_plot,
                    event_window_start, event_window_end,
                    output_path):
    """
    Multi-panel time series: T2, wind speed, pressure for selected stations.
    WRF (red) vs ASOS (blue), cold pool arrival marked as vertical green line.
    """
    n = len(stations_to_plot)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n),
                             sharex=False)
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('WRF vs ASOS: Cold Pool Passage at Houston Stations\n'
                 'Houston Derecho — 16 May 2024',
                 fontsize=14, fontweight='bold', y=1.01)

    col_titles = ['2-m Temperature (°C)', '10-m Wind Speed (m/s)',
                  'Sea-Level Pressure (mb)']
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=12, fontweight='bold', pad=8)

    for i, station in enumerate(stations_to_plot):
        df_st = asos_df[asos_df['station'] == station].copy()
        df_st = df_st.sort_values('valid')

        # WRF at this station
        meta  = asos_df[asos_df['station'] == station].iloc[0]
        wrf_ts = extract_wrf_at_station(wrf_data, meta['lat'], meta['lon'])
        wrf_times = wrf_ts['times']

        # Cold pool arrival (windowed)
        cp_time, cp_drop = detect_cold_pool(
            asos_df, station, event_window_start, event_window_end)

        # Pre-event mean pressure for anomaly
        pre_mask = df_st['valid'] <= event_window_start
        pre_pres = df_st.loc[pre_mask, 'mslp_mb'].mean()
        wrf_pre_pres = np.nanmean(wrf_ts['PSFC'][:3]) if len(wrf_ts['PSFC']) > 3 else np.nanmean(wrf_ts['PSFC'])

        for j, (obs_col, wrf_vals, ylabel) in enumerate([
            ('temp_c',  wrf_ts['T2'],   '°C'),
            ('gust_ms', wrf_ts['WSPD'], 'm/s'),
            ('mslp_mb', wrf_ts['PSFC'], 'mb'),
        ]):
            ax = axes[i, j]

            # ASOS
            ax.plot(df_st['valid'], df_st[obs_col],
                    color=C_OBS, lw=1.5, alpha=0.9, label='ASOS', zorder=3)

            # WRF
            ax.plot(wrf_times, wrf_vals,
                    color=C_WRF, lw=2.0, ls='--', alpha=0.9, label='WRF', zorder=3)

            # Cold pool arrival line
            if pd.notna(cp_time):
                ax.axvline(cp_time, color=C_CP, lw=2, ls='-',
                           alpha=0.8, label='CP arrival', zorder=4)

            # Event window shading
            ax.axvspan(event_window_start, event_window_end,
                       alpha=0.06, color='orange', zorder=1)

            ax.set_xlim(asos_df['valid'].min(), asos_df['valid'].max())
            ax.grid(True, color=C_GRID, lw=0.5, zorder=0)
            ax.tick_params(axis='x', rotation=30, labelsize=8)

            if j == 0:
                name = meta['name'] if 'name' in meta else station
                ax.set_ylabel(f'{station}\n{name}', fontsize=10, fontweight='bold')

    # Legend on last row
    legend_elements = [
        Line2D([0], [0], color=C_OBS, lw=1.5, label='ASOS observed'),
        Line2D([0], [0], color=C_WRF, lw=2.0, ls='--', label='WRF simulated'),
        Line2D([0], [0], color=C_CP,  lw=2.0, label='Cold pool arrival (obs)'),
        Patch(facecolor='orange', alpha=0.15, label='Event window'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Spatial snapshot at peak time
# ---------------------------------------------------------------------------

def plot_spatial_snapshot(wrf_data, asos_df, peak_time,
                          event_window_start, event_window_end,
                          output_path):
    """
    Spatial map at peak event time showing:
      - WRF T2 anomaly (cold pool footprint) as filled contour
      - WRF wind speed as contour lines
      - ASOS stations colored by observed gust magnitude
      - Cold pool arrival timing as station labels
    """
    # Find closest WRF time step to peak
    time_diffs = [abs((dt - peak_time).total_seconds()) for dt in wrf_data['times']]
    peak_idx   = int(np.argmin(time_diffs))
    peak_dt    = wrf_data['times'][peak_idx]

    # Compute T2 anomaly (relative to first time step in window)
    T2_K     = wrf_data['T2'][peak_idx, :, :]
    T2_base  = wrf_data['T2'][0, :, :]
    T2_anom  = (T2_K - T2_base)               # K (same as °C for anomaly)
    WSPD     = wrf_data['WSPD'][peak_idx, :, :]
    U10      = wrf_data['U10'][peak_idx, :, :]
    V10      = wrf_data['V10'][peak_idx, :, :]
    WRF_LAT  = wrf_data['lat']
    WRF_LON  = wrf_data['lon']

    # Subsample wind vectors for plotting
    skip = max(1, WRF_LAT.shape[0] // 30)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # T2 anomaly — cold pool = negative (blue), warm = positive (red)
    vmax = max(abs(np.percentile(T2_anom, 2)), abs(np.percentile(T2_anom, 98)))
    vmax = min(vmax, 15)
    cf = ax.contourf(WRF_LON, WRF_LAT, T2_anom,
                     levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', alpha=0.85, extend='both')
    cbar = plt.colorbar(cf, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('WRF 2-m Temperature Anomaly (°C)', fontsize=11)

    # Wind speed contours
    cs = ax.contour(WRF_LON, WRF_LAT, WSPD,
                    levels=[10, 15, 20, 25, 30],
                    colors='k', linewidths=0.8, alpha=0.5)
    ax.clabel(cs, fmt='%d m/s', fontsize=8)

    # Wind vectors (subsampled)
    ax.quiver(WRF_LON[::skip, ::skip], WRF_LAT[::skip, ::skip],
              U10[::skip, ::skip], V10[::skip, ::skip],
              scale=300, width=0.002, alpha=0.5, color='k', zorder=4)

    # ASOS stations
    obs_at_peak = {}
    for station in asos_df['station'].unique():
        df_st = asos_df[asos_df['station'] == station].copy()
        df_st = df_st.sort_values('valid')
        # Find obs closest to peak time
        diffs = (df_st['valid'] - peak_time).abs()
        nearest = df_st.loc[diffs.idxmin()]
        obs_at_peak[station] = nearest

    # Normalize gust for colormap
    gusts = [obs_at_peak[s]['gust_ms'] for s in obs_at_peak
             if not np.isnan(obs_at_peak[s]['gust_ms'])]
    gust_norm = mcolors.Normalize(vmin=0, vmax=max(gusts) if gusts else 30)
    gust_cmap = plt.cm.YlOrRd

    for station, row in obs_at_peak.items():
        slat = row['lat']
        slon = row['lon']
        gust = row['gust_ms']

        color = gust_cmap(gust_norm(gust)) if not np.isnan(gust) else 'gray'
        ax.scatter(slon, slat, c=[color], s=120, zorder=6,
                   edgecolors='k', linewidths=1.2)

        # Cold pool arrival label
        cp_time, _ = detect_cold_pool(asos_df, station,
                                      event_window_start, event_window_end)
        if pd.notna(cp_time):
            label = f"{station}\nCP:{cp_time.strftime('%H:%MZ')}"
        else:
            label = station
        ax.annotate(label, (slon, slat), xytext=(4, 4),
                    textcoords='offset points', fontsize=7.5,
                    fontweight='bold', color='k',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

    # ASOS gust colorbar
    sm = plt.cm.ScalarMappable(cmap=gust_cmap, norm=gust_norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, ax=ax, pad=0.08, shrink=0.5, location='right')
    cbar2.set_label('ASOS Observed Gust (m/s)', fontsize=10)

    ax.set_xlim(WRF_LON.min(), WRF_LON.max())
    ax.set_ylim(WRF_LAT.min(), WRF_LAT.max())
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'WRF Cold Pool Footprint vs ASOS Observations\n'
                 f'Houston Derecho — {peak_dt.strftime("%Y-%m-%d %H:%MZ")}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, color=C_GRID, lw=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Animated GIF — cold pool sweep
# ---------------------------------------------------------------------------

def make_coldpool_gif(wrf_data, asos_df, event_window_start, event_window_end,
                      output_path, fps=4):
    """
    Animated GIF showing WRF T2 anomaly evolving over time,
    with ASOS stations colored by observed gust at each frame.
    Stations flash green when cold pool is detected.
    """
    WRF_LAT = wrf_data['lat']
    WRF_LON = wrf_data['lon']
    T2_base = wrf_data['T2'][0, :, :]

    # Consistent color scale across all frames
    all_anom = wrf_data['T2'] - T2_base
    vmax = min(np.percentile(np.abs(all_anom), 98), 15)

    # Precompute ASOS obs nearest to each WRF timestep
    station_list = asos_df['station'].unique()
    station_meta = {s: asos_df[asos_df['station'] == s].iloc[0]
                    for s in station_list}
    cp_arrivals  = {s: detect_cold_pool(asos_df, s,
                                        event_window_start, event_window_end)[0]
                    for s in station_list}

    fig, ax = plt.subplots(figsize=(12, 9))
    skip = max(1, WRF_LAT.shape[0] // 25)

    frames = []

    for tidx, dt in enumerate(wrf_data['times']):
        ax.clear()
        T2_anom = wrf_data['T2'][tidx, :, :] - T2_base
        WSPD    = wrf_data['WSPD'][tidx, :, :]
        U10     = wrf_data['U10'][tidx, :, :]
        V10     = wrf_data['V10'][tidx, :, :]

        cf = ax.contourf(WRF_LON, WRF_LAT, T2_anom,
                         levels=np.linspace(-vmax, vmax, 21),
                         cmap='RdBu_r', alpha=0.85, extend='both')

        ax.contour(WRF_LON, WRF_LAT, WSPD,
                   levels=[15, 20, 25], colors='k',
                   linewidths=0.6, alpha=0.4)

        ax.quiver(WRF_LON[::skip, ::skip], WRF_LAT[::skip, ::skip],
                  U10[::skip, ::skip], V10[::skip, ::skip],
                  scale=350, width=0.0018, alpha=0.45, color='k')

        # ASOS stations
        for station in station_list:
            df_st  = asos_df[asos_df['station'] == station]
            diffs  = (df_st['valid'] - dt).abs()
            nearest = df_st.loc[diffs.idxmin()]
            gust   = nearest['gust_ms']
            slat   = station_meta[station]['lat']
            slon   = station_meta[station]['lon']
            cp_arr = cp_arrivals[station]

            # Green if cold pool has arrived, otherwise yellow-red scale
            if pd.notna(cp_arr) and dt >= cp_arr:
                color  = C_CP
                marker = '*'
                size   = 180
            elif not np.isnan(gust):
                norm  = min(gust / 30.0, 1.0)
                color = plt.cm.YlOrRd(norm)
                marker = 'o'
                size   = 100
            else:
                color  = 'gray'
                marker = 'o'
                size   = 80

            ax.scatter(slon, slat, c=[color], s=size, marker=marker,
                       zorder=6, edgecolors='k', linewidths=1.0)
            ax.annotate(station, (slon, slat), xytext=(3, 3),
                        textcoords='offset points', fontsize=7,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  fc='white', alpha=0.5))

        ax.set_xlim(WRF_LON.min(), WRF_LON.max())
        ax.set_ylim(WRF_LAT.min(), WRF_LAT.max())
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'WRF Cold Pool Sweep — Houston Derecho\n'
                     f'{dt.strftime("%Y-%m-%d %H:%MZ")}  '
                     f'(★ = cold pool arrived at station)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, color=C_GRID, lw=0.4, alpha=0.6)

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)

    plt.close()

    # Write GIF using matplotlib animation
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    im = ax2.imshow(frames[0])
    ax2.axis('off')

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig2, update, frames=len(frames),
                                  interval=1000//fps, blit=True)

    # Save as GIF using pillow writer
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Vertical cross section
# ---------------------------------------------------------------------------

def plot_cross_section(wrfout_files, peak_time, cross_lon,
                       event_window_start, event_window_end,
                       output_path):
    """
    N-S vertical cross section at a fixed longitude showing:
      - Temperature perturbation (cold pool depth and structure)
      - Vertical velocity (W) as contours
    at the peak event time.
    """
    col_data = extract_wrf_column(wrfout_files, 30.5, cross_lon,
                                  event_window_start, event_window_end)

    # Find time closest to peak
    diffs   = [abs((dt - peak_time).total_seconds()) for dt in col_data['times']]
    peak_idx = int(np.argmin(diffs))
    peak_dt  = col_data['times'][peak_idx]

    T_pert = col_data['T_pert'][peak_idx, :, :]   # (nz, ny)
    W      = col_data['W'][peak_idx, :, :]         # (nz, ny)
    HGT    = col_data['HGT'][peak_idx, :, :]       # (nz, ny) meters
    lats   = col_data['lats']                       # (ny,)

    # Height array — use domain mean height profile for y-axis
    mean_hgt = np.mean(HGT, axis=1)  # (nz,) mean height at each level

    # Meshgrid for plotting
    LAT_GRID = np.tile(lats, (T_pert.shape[0], 1))
    HGT_GRID = HGT

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: Temperature perturbation
    vmax_t = np.percentile(np.abs(T_pert), 98)
    vmax_t = min(vmax_t, 15)
    cf1 = axes[0].contourf(LAT_GRID, HGT_GRID, T_pert,
                           levels=np.linspace(-vmax_t, vmax_t, 21),
                           cmap='RdBu_r', extend='both')
    plt.colorbar(cf1, ax=axes[0], label='Potential Temp Perturbation (K)')

    # Cold pool top — approximate as height where T_pert < -2K
    cp_top_mask = T_pert < -2.0
    for j in range(len(lats)):
        col_mask = cp_top_mask[:, j]
        if col_mask.any():
            top_idx = np.where(col_mask)[0].max()
            axes[0].plot(lats[j], HGT_GRID[top_idx, j],
                         'k.', ms=3, alpha=0.6)

    axes[0].set_ylim(0, 5000)
    axes[0].set_ylabel('Height (m AGL)', fontsize=11)
    axes[0].set_title(f'(a) Potential Temperature Perturbation\n'
                      f'N-S Cross Section at {cross_lon:.1f}°W — '
                      f'{peak_dt.strftime("%Y-%m-%d %H:%MZ")}',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, color=C_GRID, lw=0.5)
    axes[0].annotate('Cold pool top (T pert < -2K)',
                     xy=(0.02, 0.05), xycoords='axes fraction',
                     fontsize=9, color='k',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    # Panel B: Vertical velocity
    vmax_w = max(abs(np.percentile(W, 2)), abs(np.percentile(W, 98)))
    vmax_w = min(vmax_w, 5)
    cf2 = axes[1].contourf(LAT_GRID, HGT_GRID, W,
                           levels=np.linspace(-vmax_w, vmax_w, 21),
                           cmap='PuOr_r', extend='both')
    plt.colorbar(cf2, ax=axes[1], label='Vertical Velocity W (m/s)')
    cs2 = axes[1].contour(LAT_GRID, HGT_GRID, W,
                          levels=[-1, 1], colors='k',
                          linewidths=1.0, linestyles=['--', '-'])
    axes[1].clabel(cs2, fmt='%.0f m/s', fontsize=8)

    axes[1].set_ylim(0, 5000)
    axes[1].set_xlabel('Latitude (°N)', fontsize=11)
    axes[1].set_ylabel('Height (m AGL)', fontsize=11)
    axes[1].set_title('(b) Vertical Velocity W (+ = upward)',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, color=C_GRID, lw=0.5)

    fig.suptitle('WRF Cold Pool Vertical Structure — Houston Derecho',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cold pool analysis: WRF vs ASOS')
    parser.add_argument('--event',       required=True)
    parser.add_argument('--asos',        required=True,
                        help='Path to {event}_asos_timeseries.csv')
    parser.add_argument('--wrfout',      nargs='+', required=True,
                        help='WRF output files (wrfout_d01_*)')
    parser.add_argument('--output-dir',  default='figures/coldpool')
    parser.add_argument('--peak-time',   required=True,
                        help='UTC time of peak event: "YYYY-MM-DD HH:MM"')
    parser.add_argument('--event-window-start', required=True,
                        help='Start of event window (UTC): "YYYY-MM-DD HH:MM"')
    parser.add_argument('--event-window-end', required=True,
                        help='End of event window (UTC): "YYYY-MM-DD HH:MM"')
    parser.add_argument('--cross-lon',   type=float, default=-95.5,
                        help='Longitude for N-S cross section (default: -95.5)')
    parser.add_argument('--stations',    nargs='+',
                        default=['KDWH', 'KIAH', 'KSGR', 'KHOU', 'KCLL'],
                        help='Stations to include in time series panel')
    parser.add_argument('--gif-fps',     type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peak_time           = pd.to_datetime(args.peak_time)
    event_window_start  = pd.to_datetime(args.event_window_start)
    event_window_end    = pd.to_datetime(args.event_window_end)

    # Load ASOS time series
    print("Loading ASOS time series...")
    asos_df = pd.read_csv(args.asos)
    asos_df['valid'] = pd.to_datetime(asos_df['valid'])

    # Load WRF surface fields for event window
    print("Loading WRF surface fields...")
    wrf_data = extract_wrf_surface(
        args.wrfout, event_window_start, event_window_end)

    # Figure 1: Time series
    print("\nGenerating time series figure...")
    stations_to_plot = [s for s in args.stations
                        if s in asos_df['station'].unique()]
    plot_timeseries(
        asos_df, wrf_data, stations_to_plot,
        event_window_start, event_window_end,
        output_dir / f'coldpool_timeseries_{args.event}.png')

    # Figure 2: Spatial snapshot
    print("Generating spatial snapshot figure...")
    plot_spatial_snapshot(
        wrf_data, asos_df, peak_time,
        event_window_start, event_window_end,
        output_dir / f'coldpool_spatial_{args.event}.png')

    # Figure 3: Animated GIF
    print("Generating animated GIF (this may take a few minutes)...")
    make_coldpool_gif(
        wrf_data, asos_df, event_window_start, event_window_end,
        output_dir / f'coldpool_sweep_{args.event}.gif',
        fps=args.gif_fps)

    # Figure 4: Cross section
    print("Generating vertical cross section...")
    plot_cross_section(
        args.wrfout, peak_time, args.cross_lon,
        event_window_start, event_window_end,
        output_dir / f'coldpool_crosssection_{args.event}.png')

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    main()

