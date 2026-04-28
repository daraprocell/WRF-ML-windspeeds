#!/usr/bin/env python3
"""
wind_swath_comparison.py
------------------------
Compare WRF maximum wind swath against ASOS observed peak gusts.

The figure shows:
  - Background: WRF maximum 10m wind speed at each grid point over the
    full simulation (the "wind swath") — shows WHERE WRF put its strongest winds
  - Hatching: when the WRF peak occurred at each grid point — shows WHEN
    WRF's wind maximum passed through each area
  - ASOS dots: observed peak gust at each station (same colorscale as WRF)
    with magnitude annotation — shows what was ACTUALLY observed
  - The visual contrast between bright WRF colors (northwest of Houston) and
    bright ASOS dots (over Houston) directly communicates spatial displacement

What this figure honestly communicates:
  - Magnitude: WRF wind swath vs ASOS observed (same colorscale)
  - Location: where WRF put its strongest winds vs where observations are
  - Timing: when WRF peaked across the domain (hatching) — NOT compared at
    station level because WRF's storm never passed through the station locations

Example use:
    python wind_swath_comparison.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d01_* \
        --asos data/asos/houston_asos_summary.csv \
        --output figures/wind_swath_comparison.png \
        --event-start "2024-05-16 18:00" \
        --event-end "2024-05-17 02:00"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from netCDF4 import Dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# WRF wind swath extraction
# ---------------------------------------------------------------------------

def compute_wrf_wind_swath(wrfout_files, event_start, event_end):
    """
    Compute the maximum 10m wind speed at each WRF grid point over the
    event window, and record the time at which that maximum occurred.

    Returns:
        wrf_lat   : (ny, nx) latitude array
        wrf_lon   : (ny, nx) longitude array
        max_wspd  : (ny, nx) peak wind speed (m/s) at each grid point
        time_of_max: (ny, nx) hour (UTC) when peak occurred at each grid point
    """
    files = sorted(wrfout_files)

    # Load grid from first file
    with Dataset(files[0], 'r') as nc:
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]
        ny, nx  = wrf_lat.shape

    max_wspd    = np.zeros((ny, nx))
    time_of_max = np.full((ny, nx), np.nan)

    print("Computing WRF wind swath...")
    n_loaded = 0

    for fpath in files:
        with Dataset(fpath, 'r') as nc:
            times_raw = nc.variables['Times'][:]
            ntimes    = times_raw.shape[0]

            for t in range(ntimes):
                tstr = ''.join([c.decode() for c in times_raw[t]]).replace('_', ' ')
                dt   = pd.to_datetime(tstr)

                if not (event_start <= dt <= event_end):
                    continue

                u10  = nc.variables['U10'][t, :, :]
                v10  = nc.variables['V10'][t, :, :]
                wspd = np.sqrt(u10**2 + v10**2)

                # Update max where this timestep exceeds current max
                update_mask = wspd > max_wspd
                max_wspd[update_mask]    = wspd[update_mask]
                time_of_max[update_mask] = dt.hour + dt.minute / 60.0

                n_loaded += 1

    print(f"  Processed {n_loaded} WRF time steps")
    print(f"  Domain max wind: {max_wspd.max():.1f} m/s")
    print(f"  Domain mean max: {max_wspd.mean():.1f} m/s")

    return wrf_lat, wrf_lon, max_wspd, time_of_max


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_wind_swath(wrf_lat, wrf_lon, max_wspd, time_of_max,
                    asos_summary_df, event_start, event_end,
                    output_path):
    """
    Single-panel wind swath comparison figure.
    """

    # ---- Colormap and normalization ----------------------------------------
    THRESH = 12.0   # only show winds above this threshold
    vmin, vmax = THRESH, 30
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ---- Figure setup with cartopy -----------------------------------------
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig, ax = plt.subplots(1, 1, figsize=(14, 10),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        use_cartopy = True
    except ImportError:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        use_cartopy = False
        print("  Cartopy not available — plotting without map features")

    # ---- WRF wind swath background -----------------------------------------
    swath_masked = np.where(max_wspd >= THRESH, max_wspd, np.nan)

    if use_cartopy:
        pm = ax.pcolormesh(wrf_lon, wrf_lat, swath_masked,
                           cmap=cmap, norm=norm,
                           shading='auto', alpha=0.88, zorder=2,
                           transform=ccrs.PlateCarree())
    else:
        pm = ax.pcolormesh(wrf_lon, wrf_lat, swath_masked,
                           cmap=cmap, norm=norm,
                           shading='auto', alpha=0.88, zorder=2)

    # ---- Add coastline and state borders -----------------------------------
    if use_cartopy:
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                       linewidth=1.2, edgecolor='black', zorder=5)
        ax.add_feature(cfeature.STATES.with_scale('10m'),
                       linewidth=0.7, edgecolor='#444444',
                       facecolor='none', zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                       linewidth=0.8, edgecolor='#444444', zorder=5)

    # ---- ASOS station dots -------------------------------------------------
    asos_df = asos_summary_df.copy()

    in_domain = (
        (asos_df['lat'] >= wrf_lat.min()) &
        (asos_df['lat'] <= wrf_lat.max()) &
        (asos_df['lon'] >= wrf_lon.min()) &
        (asos_df['lon'] <= wrf_lon.max())
    )
    asos_df = asos_df[in_domain].copy()

    # Manual label offsets (pixels) per station to avoid overlap
    # Format: 'STATION': (x_offset, y_offset)
    label_offsets = {
        'KHOU': (-65, -30),
        'KIAH': ( 10,  10),
        'KDWH': ( 10,  10),
        'KSGR': (-65, -30),
        'KTME': (-75,  10),
        'KCLL': (-65,  10),
        'KGLS': (  8, -30),
        'KBPT': (  8,  10),
        'KLCH': (  8,  10),
        'KLFT': (  8, -30),
        'KMSY': (-65, -30),
        'KNEW': (  8,  10),
        'KBTR': (  8,  10),
    }

    for idx, row in asos_df.iterrows():
        slat, slon = row['lat'], row['lon']

        dist = np.sqrt((wrf_lat - slat)**2 + (wrf_lon - slon)**2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        wrf_at_station = max_wspd[i, j]

        obs_gust  = row['peak_gust']
        station   = row['station']
        dot_color = cmap(norm(min(max(obs_gust, vmin), vmax))) \
                    if not np.isnan(obs_gust) else 'gray'

        ax.scatter(slon, slat, c=[dot_color], s=220, zorder=8,
                   edgecolors='black', linewidths=2.0, marker='o',
                   **({'transform': ccrs.PlateCarree()} if use_cartopy else {}))

        # Compact single-line label
        label_text = f"{station}: {obs_gust:.0f}/{wrf_at_station:.0f} m/s"
        xoff, yoff = label_offsets.get(station, (8, 6))

        ax.annotate(label_text, xy=(slon, slat),
                    xytext=(xoff, yoff), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec='gray', alpha=0.85, lw=0.7),
                    arrowprops=dict(arrowstyle='-', color='gray',
                                   lw=0.8, alpha=0.6),
                    **({'xycoords': ccrs.PlateCarree()._as_mpl_transform(ax)}
                       if use_cartopy else {}))

    # ---- Downtown Houston marker -------------------------------------------
    ax.scatter(-95.35, 29.75, c='black', s=180, zorder=10,
               marker='*', edgecolors='white', linewidths=1.5,
               **({'transform': ccrs.PlateCarree()} if use_cartopy else {}))
    ax.annotate('Downtown\nHouston', xy=(-95.35, 29.75),
                xytext=(6, -22), textcoords='offset points',
                fontsize=9, fontweight='bold', color='black', zorder=11,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85),
                **({'xycoords': ccrs.PlateCarree()._as_mpl_transform(ax)}
                   if use_cartopy else {}))

    # ---- WRF domain-wide peak location -------------------------------------
    peak_idx      = np.unravel_index(np.argmax(max_wspd), max_wspd.shape)
    peak_lat      = wrf_lat[peak_idx]
    peak_lon      = wrf_lon[peak_idx]
    peak_val      = max_wspd[peak_idx]
    peak_time_hr  = time_of_max[peak_idx]
    peak_time_str = f"{int(peak_time_hr):02d}:{int((peak_time_hr % 1)*60):02d}Z"

    ax.scatter(peak_lon, peak_lat, c='white', s=300, zorder=10,
               edgecolors='black', linewidths=2.5, marker='*',
               **({'transform': ccrs.PlateCarree()} if use_cartopy else {}))
    ax.annotate(f"WRF domain max\n{peak_val:.1f} m/s @ {peak_time_str}",
                xy=(peak_lon, peak_lat),
                xytext=(10, -25), textcoords='offset points',
                fontsize=8.5, fontweight='bold', zorder=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                          ec='black', alpha=0.9, lw=1.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                **({'xycoords': ccrs.PlateCarree()._as_mpl_transform(ax)}
                   if use_cartopy else {}))

    # ---- Colorbar ----------------------------------------------------------
    cbar = plt.colorbar(pm, ax=ax, pad=0.02, shrink=0.85,
                        extend='min')
    cbar.set_label(f'Peak 10-m Wind Speed (m/s, ≥{THRESH:.0f} m/s shown)\n'
                   '[WRF swath = background  |  ASOS = filled circles]',
                   fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ---- Legend ------------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='tomato', markeredgecolor='black',
               markersize=12, markeredgewidth=2,
               label='ASOS station — label: Obs/WRF peak (m/s)'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='black', markeredgecolor='white',
               markersize=14, markeredgewidth=1.5,
               label='Downtown Houston'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='white', markeredgecolor='black',
               markersize=14, markeredgewidth=2,
               label='WRF domain-wide wind maximum'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=9, framealpha=0.92,
              title='Legend', title_fontsize=9.5)

    # ---- Map bounds and formatting -----------------------------------------
    if use_cartopy:
        ax.set_extent([wrf_lon.min(), wrf_lon.max(),
                       wrf_lat.min(), wrf_lat.max()],
                      crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.4,
                     color='gray', alpha=0.5, linestyle='--')
    else:
        ax.set_xlim(wrf_lon.min(), wrf_lon.max())
        ax.set_ylim(wrf_lat.min(), wrf_lat.max())
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.grid(True, color='gray', lw=0.4, alpha=0.4)

    ax.set_title(
        'WRF Simulated Maximum Wind Swath vs ASOS Observed Peak Gusts\n'
        'Houston Derecho — 16 May 2024   '
        f'(Event window: {event_start.strftime("%H:%MZ")} – '
        f'{event_end.strftime("%H:%MZ")})\n'
        f'Background = WRF peak wind at each grid point (≥{THRESH:.0f} m/s)  |  '
        'Circles = ASOS observed peak gust  |  '
        'Station labels = Obs/WRF (m/s)',
        fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # ---- Print summary table -----------------------------------------------
    print("\n--- Wind comparison summary ---")
    print(f"{'Station':8s} | {'Obs gust':10s} | {'WRF at stn':12s} | "
          f"{'Difference':12s}")
    print('-' * 52)
    for _, row in asos_df.iterrows():
        slat, slon = row['lat'], row['lon']
        dist = np.sqrt((wrf_lat - slat)**2 + (wrf_lon - slon)**2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        wrf_val = max_wspd[i, j]
        obs_val = row['peak_gust']
        diff    = wrf_val - obs_val
        print(f"{row['station']:8s} | {obs_val:8.1f} m/s | "
              f"{wrf_val:10.1f} m/s | {diff:+.1f} m/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='WRF wind swath vs ASOS comparison')
    parser.add_argument('--wrfout',      nargs='+', required=True,
                        help='WRF output files (wrfout_d01_* or wrfout_d02_*)')
    parser.add_argument('--asos',        required=True,
                        help='ASOS summary CSV (houston_asos_summary.csv)')
    parser.add_argument('--output',      default='figures/wind_swath_comparison.png')
    parser.add_argument('--event-start', default='2024-05-16 18:00',
                        help='Start of event window (UTC)')
    parser.add_argument('--event-end',   default='2024-05-17 02:00',
                        help='End of event window (UTC)')
    args = parser.parse_args()

    event_start = pd.to_datetime(args.event_start)
    event_end   = pd.to_datetime(args.event_end)

    # Load ASOS summary
    print("Loading ASOS summary...")
    asos_df = pd.read_csv(args.asos)
    print(f"  {len(asos_df)} stations loaded")

    # Compute WRF wind swath
    wrf_lat, wrf_lon, max_wspd, time_of_max = compute_wrf_wind_swath(
        args.wrfout, event_start, event_end)

    # Plot
    print("\nGenerating figure...")
    plot_wind_swath(
        wrf_lat, wrf_lon, max_wspd, time_of_max,
        asos_df, event_start, event_end,
        args.output
    )


if __name__ == '__main__':
    main()