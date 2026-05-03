#!/usr/bin/env python3
"""
crosssection.py
-------------------
N-S vertical cross section through the WRF derecho cold pool.

Variables used:
  T    : WRF perturbation potential temperature (K)
  P    : WRF perturbation pressure (Pa)
  PB   : WRF base pressure (Pa)
  PH   : WRF perturbation geopotential (m^2/s^2)
  PHB  : WRF base geopotential (m^2/s^2)
  U    : WRF u-wind component (m/s, staggered west-east)
  W    : WRF vertical velocity (m/s, staggered bottom-top)

Temperature calculation:
  theta = T + 300                          (full potential temperature, K)
  T_K   = theta * (P_total/100000)^0.286  (actual temperature, K)
  T_anom = T_K - T_K_baseline             (anomaly relative to pre-storm mean)

Example use:
    python crosssection.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d02_* \
        --peak-time "2024-05-17 00:00" \
        --cross-lon -95.35 \
        --baseline-end "2024-05-16 12:00" \
        --output figures/crosssection/crosssection_houston.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

R_cp = 0.2854   # R/cp for dry air
P0   = 100000.0 # reference pressure (Pa)


def load_wrf_times(wrfout_files):
    """Return sorted list of (filepath, time_index, datetime) tuples."""
    entries = []
    for fpath in sorted(wrfout_files):
        with Dataset(fpath, 'r') as nc:
            times_raw = nc.variables['Times'][:]
            for t in range(times_raw.shape[0]):
                tstr = ''.join([c.decode() for c in times_raw[t]]).replace('_', ' ')
                dt = pd.to_datetime(tstr)
                entries.append((fpath, t, dt))
    return sorted(entries, key=lambda x: x[2])


def find_cross_section_column(wrfout_files, cross_lon):
    """
    Find the WRF grid column index closest to cross_lon,
    and return the lat array along that column.
    """
    with Dataset(sorted(wrfout_files)[0], 'r') as nc:
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]

    # Use middle row to find column index
    mid_row = wrf_lat.shape[0] // 2
    col_idx = int(np.argmin(np.abs(wrf_lon[mid_row, :] - cross_lon)))
    lats    = wrf_lat[:, col_idx]
    actual_lon = wrf_lon[mid_row, col_idx]

    return col_idx, lats


def compute_actual_temperature(T_pert, P_pert, PB):
    """
    Convert WRF perturbation potential temperature to actual temperature (K).

    T_pert : perturbation potential temperature (K), WRF variable 'T'
    P_pert : perturbation pressure (Pa), WRF variable 'P'
    PB     : base pressure (Pa), WRF variable 'PB'

    Returns actual temperature T (K).
    """
    theta     = T_pert + 300.0          # full potential temperature (K)
    P_total   = P_pert + PB             # total pressure (Pa)
    T_actual  = theta * (P_total / P0) ** R_cp  # actual temperature (K)
    return T_actual


def extract_cross_section(all_times, col_idx, baseline_end):
    """
    Extract vertical cross section data at a fixed column for all time steps.

    Returns dict with arrays shaped (ntime, nz, ny):
      T_K    : actual temperature (K)
      U      : zonal wind (m/s, unstaggered)
      W      : vertical velocity (m/s, unstaggered)
      HGT    : height AGL (m, unstaggered)
    """
    T_all   = []
    U_all   = []
    W_all   = []
    HGT_all = []
    times   = []

    for fpath, tidx, dt in all_times:
        with Dataset(fpath, 'r') as nc:
            # WRF perturbation potential temp (nz, ny)
            T_pert = nc.variables['T'][tidx, :, :, col_idx]

            # Pressure
            P_pert = nc.variables['P'][tidx, :, :, col_idx]
            PB     = nc.variables['PB'][tidx, :, :, col_idx]

            # Geopotential height 
            PH     = nc.variables['PH'][tidx, :, :, col_idx]
            PHB    = nc.variables['PHB'][tidx, :, :, col_idx]

            # U wind
            U_stag = nc.variables['U'][tidx, :, :, col_idx:col_idx+2]
            U      = 0.5 * (U_stag[:, :, 0] + U_stag[:, :, 1])

            # Vertical velocity 
            W_stag = nc.variables['W'][tidx, :, :, col_idx]

        T_K = compute_actual_temperature(T_pert, P_pert, PB)

        # Unstagger geopotential height
        HGT_stag = (PH + PHB) / 9.81  # m
        HGT      = 0.5 * (HGT_stag[:-1, :] + HGT_stag[1:, :])

        # Unstagger W
        W = 0.5 * (W_stag[:-1, :] + W_stag[1:, :])

        # Subtract terrain height to get AGL
        terrain = HGT[0, :]  # approximate terrain as lowest level height
        HGT_AGL = HGT - terrain[np.newaxis, :]

        T_all.append(T_K)
        U_all.append(U)
        W_all.append(W)
        HGT_all.append(HGT_AGL)
        times.append(dt)

    # Compute temperature baseline: mean over all times before baseline_end
    T_arr      = np.array(T_all)
    baseline_mask = np.array([dt <= baseline_end for dt in times])
    if baseline_mask.sum() == 0:
        print("WARNING: No baseline times found, using first timestep as baseline")
        T_baseline = T_arr[0:1, :, :].mean(axis=0)
    else:
        T_baseline = T_arr[baseline_mask, :, :].mean(axis=0)

    return {
        'times':     times,
        'T_K':       T_arr,
        'T_anom':    T_arr - T_baseline[np.newaxis, :, :],  # anomaly (K)
        'T_baseline': T_baseline,
        'U':         np.array(U_all),
        'W':         np.array(W_all),
        'HGT':       np.array(HGT_all),
    }


def plot_cross_section(data, lats, peak_time, cross_lon, output_path,
                       max_height=5000, houston_lat=29.75):
    """
    Three-panel cross section with Houston metro highlighted:
      (a) Temperature anomaly — cold pool depth and intensity
      (b) Horizontal wind speed U — shows rear-inflow jet and surface wind deficit,
          with the inverted-profile zone over Houston explicitly annotated
      (c) Vertical velocity W — updraft/downdraft structure
    """
    # Find peak time index
    diffs    = [abs((dt - peak_time).total_seconds()) for dt in data['times']]
    peak_idx = int(np.argmin(diffs))
    peak_dt  = data['times'][peak_idx]

    T_anom = data['T_anom'][peak_idx, :, :]   # (nz, ny)
    U      = data['U'][peak_idx, :, :]         # (nz, ny)
    W      = data['W'][peak_idx, :, :]         # (nz, ny)
    HGT    = data['HGT'][peak_idx, :, :]       # (nz, ny) AGL

    # Find lat-index nearest to Houston for diagnostics
    hou_j = int(np.argmin(np.abs(lats - houston_lat)))

    # Mask above max_height
    height_mask = HGT <= max_height

    # Build lat/height grids for plotting
    LAT_GRID = np.tile(lats, (T_anom.shape[0], 1))

    # Wind speed magnitude (just U for N-S cross section)
    WSPD = np.abs(U)

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    plt.rcParams.update({'font.size': 11})

    # Panel A: Temp anomaly
    vmax_t = min(np.nanpercentile(np.abs(T_anom[height_mask]), 98), 12)

    cf1 = axes[0].contourf(LAT_GRID, HGT, T_anom,
                           levels=np.linspace(-vmax_t, vmax_t, 25),
                           cmap='RdBu_r', extend='both')
    plt.colorbar(cf1, ax=axes[0], label='Temperature Anomaly (K)',
                 pad=0.02, shrink=0.95)

    # Cold pool top, where T anomaly < -2K
    cp_top_mask = T_anom < -2.0
    for j in range(len(lats)):
        col = cp_top_mask[:, j]
        valid = col & (HGT[:, j] <= max_height)
        if valid.any():
            top_idx = np.where(valid)[0].max()
            axes[0].plot(lats[j], HGT[top_idx, j],
                         'k.', ms=2.5, alpha=0.7)

    # Contour the -2K line 
    axes[0].contour(LAT_GRID, HGT, T_anom,
                    levels=[-2], colors='navy',
                    linewidths=1.5, linestyles='--')

    axes[0].set_ylim(0, max_height)
    axes[0].set_ylabel('Height AGL (m)', fontsize=11)
    axes[0].set_title(
        f'(a) Temperature Anomaly relative to pre-storm baseline\n'
        f'N-S Cross Section at {abs(cross_lon):.2f}\u00b0W (Houston longitude) \u2014 '
        f'{peak_dt.strftime("%Y-%m-%d %H:%MZ")}',
        fontsize=11, fontweight='bold')
    axes[0].grid(True, color='#DDDDDD', lw=0.5)
    axes[0].annotate('Cold pool top (\u22122K contour = dashed navy)',
                     xy=(0.02, 0.06), xycoords='axes fraction',
                     fontsize=8.5, color='navy',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # Mark Houston latitude on all panels
    axes[0].axvline(houston_lat, color='black', lw=1.5, ls='-', alpha=0.7,
                    zorder=10)
    axes[0].annotate('Houston\nmetro',
                     xy=(houston_lat, max_height * 0.92),
                     xytext=(3, 0), textcoords='offset points',
                     fontsize=9, fontweight='bold',
                     ha='left', va='center', zorder=11,
                     bbox=dict(boxstyle='round,pad=0.25',
                               fc='yellow', ec='black', alpha=0.9, lw=0.8))

    # Panel B: Horizontal wind speed (U)
    # Shows the rear inflow jet aloft vs weak surface winds
    wspd_levels = np.arange(0, 30, 2)
    cf2 = axes[1].contourf(LAT_GRID, HGT, WSPD,
                           levels=wspd_levels,
                           cmap='YlOrRd', extend='max')
    plt.colorbar(cf2, ax=axes[1], label='|U| Wind Speed (m/s)',
                 pad=0.02, shrink=0.95)

    cs2 = axes[1].contour(LAT_GRID, HGT, WSPD,
                          levels=[10, 15, 20],
                          colors='k', linewidths=0.8, alpha=0.6)
    axes[1].clabel(cs2, fmt='%d m/s', fontsize=8)

    axes[1].set_ylim(0, max_height)
    axes[1].set_ylabel('Height AGL (m)', fontsize=11)
    axes[1].set_title('(b) Horizontal Wind Speed |U| \u2014 '
                      'Inverted Profile Over Houston (PBL Transport Failure)',
                      fontsize=11, fontweight='bold')
    axes[1].grid(True, color='#DDDDDD', lw=0.5)

    # Mark Houston longitude on panel (b)
    axes[1].axvline(houston_lat, color='black', lw=1.5, ls='-', alpha=0.7,
                    zorder=10)

    # Annotate the inverted profile zone directly over Houston:
    # box from ~Houston lat \u00b1 0.3\u00b0 spanning surface (0 m) to ~3 km (jet level)
    box_w = 0.4   # +/- degrees lat around Houston
    box_left  = houston_lat - box_w
    box_right = houston_lat + box_w
    box_top_height = 3500  # m AGL where the elevated jet sits

    # Use a Rectangle patch
    from matplotlib.patches import Rectangle
    rect = Rectangle((box_left, 0), 2 * box_w, box_top_height,
                     fill=False, edgecolor='black', lw=2.0,
                     ls='--', zorder=11, alpha=0.85)
    axes[1].add_patch(rect)

    # Diagnose surface vs aloft wind in this column for label
    col_U  = WSPD[:, hou_j]   # (nz,)
    col_H  = HGT[:, hou_j]
    surf_mask  = col_H < 200
    aloft_mask = (col_H >= 1500) & (col_H <= 3500)
    surf_U  = col_U[surf_mask].mean() if surf_mask.any() else float('nan')
    aloft_U = col_U[aloft_mask].mean() if aloft_mask.any() else float('nan')

    # Annotation text: place INSIDE the dashed box, near the top so it doesn't
    # overlap the 10 m/s contour line. Box is from box_left to box_right and
    # height 0 to box_top_height; place at upper-left interior of box.
    axes[1].annotate(
        f"Inverted profile over Houston:\n"
        f"  ~{aloft_U:.0f} m/s aloft (1.5\u20133.5 km AGL)\n"
        f"  ~{surf_U:.0f} m/s at surface\n"
        f"  \u2192 PBL/sfc layer fails to transport momentum",
        xy=(box_left + 0.05, box_top_height - 200),
        ha='left', va='top',
        fontsize=8.5, fontweight='bold',
        zorder=12,
        bbox=dict(boxstyle='round,pad=0.4', fc='white',
                  ec='black', lw=1.0, alpha=0.95))

    axes[1].axhline(10, color='blue', lw=1.5, ls=':', alpha=0.8,
                    label='10m AGL (U10 level)')
    axes[1].axhline(70, color='purple', lw=1.5, ls='--', alpha=0.8,
                    label='~70m AGL (lowest model level)')
    axes[1].legend(fontsize=8, loc='upper right')

    # Panel C: Vertical velocity (W)
    vmax_w = min(max(abs(np.nanpercentile(W[height_mask], 1)),
                     abs(np.nanpercentile(W[height_mask], 99))), 5)

    cf3 = axes[2].contourf(LAT_GRID, HGT, W,
                           levels=np.linspace(-vmax_w, vmax_w, 25),
                           cmap='PuOr_r', extend='both')
    plt.colorbar(cf3, ax=axes[2], label='Vertical Velocity W (m/s)',
                 pad=0.02, shrink=0.95)

    cs3 = axes[2].contour(LAT_GRID, HGT, W,
                          levels=[-1, 1], colors='k',
                          linewidths=1.0, linestyles=['--', '-'])
    axes[2].clabel(cs3, fmt='%+.0f m/s', fontsize=8)

    axes[2].set_ylim(0, max_height)
    axes[2].set_xlabel('Latitude (\u00b0N)', fontsize=11)
    axes[2].set_ylabel('Height AGL (m)', fontsize=11)
    axes[2].set_title('(c) Vertical Velocity W  '
                      '(orange = upward, purple = downward)',
                      fontsize=11, fontweight='bold')
    axes[2].grid(True, color='#DDDDDD', lw=0.5)

    # Mark Houston latitude on panel (c)
    axes[2].axvline(houston_lat, color='black', lw=1.5, ls='-', alpha=0.7,
                    zorder=10)

    fig.suptitle('WRF Cold Pool Vertical Structure — Houston Derecho 16 May 2024',
                 fontsize=13, fontweight='bold', y=1.005)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print("Diagnostics at peak time:", peak_dt)
    print(f"Cold pool max T anomaly: {T_anom[HGT < 2000].min():.1f} K")
    cold_mask = (T_anom < -2) & (HGT < max_height)
    if cold_mask.any():
        cp_depth = HGT[cold_mask].max()
        print(f"  Cold pool max depth: {cp_depth:.0f} m AGL")
    else:
        print("  Cold pool (< -2K) not found in cross section")
    print(f"  Max |U| in cross section: {WSPD[HGT < max_height].max():.1f} m/s")
    print(f"  Max |U| below 200m AGL:  {WSPD[HGT < 200].max():.1f} m/s")
    print(f"  Max |W| in cross section: {np.abs(W[HGT < max_height]).max():.1f} m/s")

    # Houston column diagnostics
    print()
    print(f"--- Column directly over Houston (lat = {lats[hou_j]:.2f}\u00b0N) ---")
    col_U = WSPD[:, hou_j]
    col_H = HGT[:, hou_j]
    col_T = T_anom[:, hou_j]

    bands = [
        ("Surface (z < 100 m)",        col_H < 100),
        ("Near-surface (100-500 m)",   (col_H >= 100) & (col_H < 500)),
        ("Lower BL (500-1500 m)",      (col_H >= 500) & (col_H < 1500)),
        ("Mid (1500-3000 m)",          (col_H >= 1500) & (col_H < 3000)),
        ("Upper (3000-5000 m)",        (col_H >= 3000) & (col_H < 5000)),
    ]
    for label, b_mask in bands:
        if b_mask.any():
            mean_U = col_U[b_mask].mean()
            max_U  = col_U[b_mask].max()
            mean_T = col_T[b_mask].mean()
            print(f"  {label:30s}: |U| mean={mean_U:5.1f} m/s, "
                  f"max={max_U:5.1f} m/s, T_anom mean={mean_T:+5.2f} K")

    if (col_H < 100).any() and ((col_H >= 3000) & (col_H < 5000)).any():
        ratio = col_U[(col_H >= 3000) & (col_H < 5000)].mean() / col_U[col_H < 100].mean()
        print(f"\n  Surface vs aloft (3-5 km) wind speed ratio: {ratio:.1f}x reduction")


def main():
    parser = argparse.ArgumentParser(
        description='WRF cold pool cross section over Houston')
    parser.add_argument('--wrfout', nargs='+', required=True,
                        help='WRF output files')
    parser.add_argument('--peak-time', default='2024-05-17 00:00',
                        help='UTC peak time: "YYYY-MM-DD HH:MM" (default: 2024-05-17 00:00)')
    parser.add_argument('--cross-lon', type=float, default=-95.35,
                        help='Longitude for N-S cross section (default: -95.35, downtown Houston)')
    parser.add_argument('--houston-lat', type=float, default=29.75,
                        help='Houston latitude for marker (default: 29.75)')
    parser.add_argument('--baseline-end', default='2024-05-16 12:00',
                        help='End time for pre-storm baseline (default: 2024-05-16 12:00)')
    parser.add_argument('--output', default='figures/coldpool/CP_crosssection_houston.png')
    parser.add_argument('--max-height', type=int, default=5000)
    args = parser.parse_args()

    peak_time    = pd.to_datetime(args.peak_time)
    baseline_end = pd.to_datetime(args.baseline_end)

    all_times = load_wrf_times(args.wrfout)
    col_idx, lats = find_cross_section_column(args.wrfout, args.cross_lon)
    data = extract_cross_section(all_times, col_idx, baseline_end)

    plot_cross_section(data, lats, peak_time, args.cross_lon,
                       args.output, args.max_height,
                       houston_lat=args.houston_lat)


if __name__ == '__main__':
    main()