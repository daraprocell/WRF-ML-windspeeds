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
    python windswath.py \
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

    with Dataset(files[0], 'r') as nc:
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]
        ny, nx  = wrf_lat.shape

    max_wspd    = np.zeros((ny, nx))
    time_of_max = np.full((ny, nx), np.nan)

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

    print(f"Domain max wind: {max_wspd.max():.1f} m/s")
    print(f"Domain mean max: {max_wspd.mean():.1f} m/s")

    return wrf_lat, wrf_lon, max_wspd, time_of_max


def plot_wind_swath(wrf_lat, wrf_lon, max_wspd, time_of_max,
                    asos_summary_df, event_start, event_end,
                    output_path):
    """
    Single-panel wind swath comparison figure.
    """

    # Shared scale for WRF swath and ASOS dots
    vmin, vmax = 0, 30
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Divide event window into 2 hr bins
    # Each bin gets a different hatch pattern on the WRF swath
    timing_bins = [
        (18, 20, '....', '18-20Z', '#4444AA'),
        (20, 22, 'xxxx', '20-22Z', '#2288CC'),
        (22, 24, '////',  '22-00Z', '#22AA66'),
        (0,   2, '\\\\\\\\', '00-02Z', '#AA6622'),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Only show grid points above a threshold to avoid cluttering with
    # background noise.show wind swath where WRF peaked above 8 m/s
    swath_masked = np.where(max_wspd >= 8, max_wspd, np.nan)

    pm = ax.pcolormesh(wrf_lon, wrf_lat, swath_masked,
                       cmap=cmap, norm=norm,
                       shading='auto', alpha=0.85, zorder=2)

    # For each timing bin, create a filled contour with hatching
    # showing WHEN the WRF peak occurred at each grid point
    for t_start, t_end, hatch, label, color in timing_bins:

        # Handle midnight
        if t_start < t_end:
            timing_mask = (
                (time_of_max >= t_start) &
                (time_of_max <  t_end) &
                (max_wspd >= 12)        # only show timing for significant winds
            ).astype(float)
        else:
            timing_mask = (
                ((time_of_max >= t_start) | (time_of_max < t_end)) &
                (max_wspd >= 12)
            ).astype(float)

        if timing_mask.sum() > 0:
            ax.contourf(wrf_lon, wrf_lat, timing_mask,
                        levels=[0.5, 1.5],
                        hatches=[hatch],
                        colors='none',
                        alpha=0.0,
                        zorder=3)
            # Solid contour outline for the timing region
            ax.contour(wrf_lon, wrf_lat, timing_mask,
                       levels=[0.5],
                       colors=[color],
                       linewidths=1.2,
                       alpha=0.7,
                       zorder=4)

    asos_df = asos_summary_df.copy()

    # Filter to stations within the WRF domain
    in_domain = (
        (asos_df['lat'] >= wrf_lat.min()) &
        (asos_df['lat'] <= wrf_lat.max()) &
        (asos_df['lon'] >= wrf_lon.min()) &
        (asos_df['lon'] <= wrf_lon.max())
    )
    asos_df = asos_df[in_domain].copy()

    # Also extract WRF wind at each station grid point for annotation
    for idx, row in asos_df.iterrows():
        slat, slon = row['lat'], row['lon']

        # Find nearest WRF grid point
        dist = np.sqrt((wrf_lat - slat)**2 + (wrf_lon - slon)**2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        wrf_at_station = max_wspd[i, j]

        obs_gust = row['peak_gust']
        station  = row['station']

        # Dot color = observed gust (same colorscale as WRF)
        dot_color = cmap(norm(obs_gust)) if not np.isnan(obs_gust) else 'gray'

        # Plot station dot
        ax.scatter(slon, slat,
                   c=[dot_color],
                   s=220,
                   zorder=8,
                   edgecolors='black',
                   linewidths=2.0,
                   marker='o')

        # Annotation: station name, observed gust, WRF at that point
        # Two-line label: obs gust on top, WRF at station below
        label_text = (
            f"{station}\n"
            f"Obs: {obs_gust:.1f} m/s\n"
            f"WRF: {wrf_at_station:.1f} m/s"
        )

        ax.annotate(
            label_text,
            xy=(slon, slat),
            xytext=(8, 6),
            textcoords='offset points',
            fontsize=7.5,
            fontweight='bold',
            zorder=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='gray', alpha=0.85, lw=0.8),
        )

    peak_idx = np.unravel_index(np.argmax(max_wspd), max_wspd.shape)
    peak_lat = wrf_lat[peak_idx]
    peak_lon = wrf_lon[peak_idx]
    peak_val = max_wspd[peak_idx]
    peak_time_hr = time_of_max[peak_idx]
    peak_time_str = f"{int(peak_time_hr):02d}:{int((peak_time_hr % 1)*60):02d}Z"

    ax.scatter(peak_lon, peak_lat,
               c='white', s=300, zorder=10,
               edgecolors='black', linewidths=2.5,
               marker='*')
    ax.annotate(
        f"WRF domain max\n{peak_val:.1f} m/s @ {peak_time_str}",
        xy=(peak_lon, peak_lat),
        xytext=(10, -20),
        textcoords='offset points',
        fontsize=8.5,
        fontweight='bold',
        color='black',
        zorder=11,
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                  ec='black', alpha=0.9, lw=1.0),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
    )

    ax.scatter(-95.39, 29.69,
               c='black', s=120, zorder=10,
               marker='*', edgecolors='white', linewidths=1.0)
    ax.annotate('Houston', xy=(-95.39, 29.69),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold', color='black', zorder=11)

    cbar = plt.colorbar(pm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Peak 10-m Wind Speed (m/s)\n'
                   '[WRF = background swath, ASOS = filled circles]',
                   fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='gray', markeredgecolor='black',
               markersize=12, markeredgewidth=2,
               label='ASOS station (color = observed peak gust)'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='white', markeredgecolor='black',
               markersize=14, markeredgewidth=2,
               label='WRF domain-wide wind maximum'),
        mpatches.Patch(facecolor='#4444AA', alpha=0.7, label='WRF peak: 18-20Z'),
        mpatches.Patch(facecolor='#2288CC', alpha=0.7, label='WRF peak: 20-22Z'),
        mpatches.Patch(facecolor='#22AA66', alpha=0.7, label='WRF peak: 22-00Z'),
        mpatches.Patch(facecolor='#AA6622', alpha=0.7, label='WRF peak: 00-02Z'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=8.5, framealpha=0.9,
              title='Legend', title_fontsize=9)

    ax.set_xlim(wrf_lon.min(), wrf_lon.max())
    ax.set_ylim(wrf_lat.min(), wrf_lat.max())
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, color='gray', lw=0.4, alpha=0.4, zorder=1)

    ax.set_title(
        'WRF Simulated Maximum Wind Swath vs ASOS Observed Peak Gusts\n'
        'Houston Derecho — 16 May 2024   '
        f'(Event window: {event_start.strftime("%H:%MZ")} – '
        f'{event_end.strftime("%H:%MZ")})\n'
        'Background color = WRF peak at each grid point   |   '
        'Circles = ASOS observed peak gust   |   '
        'Contour outlines = timing of WRF wind maximum',
        fontsize=11, fontweight='bold', pad=10
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


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


    asos_df = pd.read_csv(args.asos)

    wrf_lat, wrf_lon, max_wspd, time_of_max = compute_wrf_wind_swath(
        args.wrfout, event_start, event_end)

    # Plot
    plot_wind_swath(
        wrf_lat, wrf_lon, max_wspd, time_of_max,
        asos_df, event_start, event_end,
        args.output
    )


if __name__ == '__main__':
    main()