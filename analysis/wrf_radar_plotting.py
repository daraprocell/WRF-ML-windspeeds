#!/usr/bin/env python3
"""
wrf_radar_comparison.py
-----------------------
Compare WRF simulated reflectivity (REFL_10CM) against IEM NEXRAD composite
reflectivity for the Houston derecho event.

Produces:
  1. wrf_refl_vs_obs_{time}.png  — side-by-side comparison at peak time
  2. wrf_refl_animation.gif      — WRF reflectivity evolution over event window
  3. obs_refl_animation.gif      — observed reflectivity over same window
  4. refl_comparison.gif         — side-by-side WRF vs obs animated GIF

Example use:
    python wrf_radar_comparison.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d01_* \
        --output-dir figures/radar \
        --event-window-start "2024-05-16 18:00" \
        --event-window-end "2024-05-17 02:00" \
        --peak-time "2024-05-16 23:00"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.patches import Patch
import requests
from io import BytesIO
from PIL import Image
from netCDF4 import Dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Radar colormap — matches NWS standard reflectivity colors
# ---------------------------------------------------------------------------

def make_nws_refl_cmap():
    """
    NWS-style reflectivity colormap (dBZ).
    Matches the standard NWS radar color scale.
    """
    colors = [
        (0.00, (0.40, 0.40, 0.40)),   # < 5 dBZ  gray
        (0.10, (0.54, 0.54, 0.54)),   # 5        light gray
        (0.17, (0.00, 0.93, 0.93)),   # 10       light blue
        (0.23, (0.00, 0.00, 0.93)),   # 15       blue
        (0.30, (0.00, 1.00, 0.00)),   # 20       green
        (0.37, (0.00, 0.79, 0.00)),   # 25       medium green
        (0.43, (0.00, 0.56, 0.00)),   # 30       dark green
        (0.50, (1.00, 1.00, 0.00)),   # 35       yellow
        (0.57, (0.93, 0.75, 0.00)),   # 40       dark yellow
        (0.63, (1.00, 0.56, 0.00)),   # 45       orange
        (0.70, (1.00, 0.00, 0.00)),   # 50       red
        (0.77, (0.84, 0.00, 0.00)),   # 55       dark red
        (0.83, (0.75, 0.00, 0.00)),   # 60       darker red
        (0.90, (1.00, 0.00, 1.00)),   # 65       magenta
        (1.00, (0.60, 0.00, 0.60)),   # 70+      dark magenta
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'nws_refl',
        [(v, c) for v, c in colors]
    )
    return cmap

NWS_CMAP  = make_nws_refl_cmap()
REFL_NORM = mcolors.Normalize(vmin=0, vmax=75)

# ASOS stations for overlaying
HOUSTON_STATIONS = {
    'KHOU': (29.6375, -95.2824),
    'KIAH':  (29.9844, -95.3607),
    'KDWH':  (30.068,  -95.5562),
    'KSGR':  (29.622,  -95.657),
    'KCLL':  (30.588,  -96.364),
    'KGLS':  (29.2653, -94.8604),
    'KBPT':  (30.0686, -94.0207),
}

# ---------------------------------------------------------------------------
# WRF loading
# ---------------------------------------------------------------------------

def load_wrf_times(wrfout_files):
    """Return list of (filepath, time_index, datetime) tuples."""
    entries = []
    for fpath in sorted(wrfout_files):
        with Dataset(fpath, 'r') as nc:
            times_raw = nc.variables['Times'][:]
            for t in range(times_raw.shape[0]):
                tstr = ''.join([c.decode() for c in times_raw[t]]).replace('_', ' ')
                dt = pd.to_datetime(tstr)
                entries.append((fpath, t, dt))
    return entries


def extract_wrf_refl(fpath, tidx):
    """
    Extract column-maximum simulated reflectivity from a WRF output file.
    Uses REFL_10CM (3D) and takes the column maximum (composite reflectivity).
    """
    with Dataset(fpath, 'r') as nc:
        refl_3d = nc.variables['REFL_10CM'][tidx, :, :, :]  # (nz, ny, nx)
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]

    # Column maximum = composite reflectivity
    refl_comp = np.max(refl_3d, axis=0)

    # Mask negative values (clear air)
    refl_comp = np.where(refl_comp < 0, np.nan, refl_comp)

    return refl_comp, wrf_lat, wrf_lon


# ---------------------------------------------------------------------------
# IEM observed radar download
# ---------------------------------------------------------------------------

def download_iem_radar(dt):
    """
    Download IEM NEXRAD composite reflectivity PNG for a given UTC datetime.
    Returns PIL Image or None if unavailable.

    IEM files are named n0q_YYYYMMDDHHММ.png at 5-minute intervals.
    We find the nearest 5-minute timestamp.
    """
    # Round to nearest 5 minutes
    minute = (dt.minute // 5) * 5
    dt_rounded = dt.replace(minute=minute, second=0, microsecond=0)

    url = (f"https://mesonet.agron.iastate.edu/archive/data/"
           f"{dt_rounded.strftime('%Y/%m/%d')}/GIS/uscomp/"
           f"n0q_{dt_rounded.strftime('%Y%m%d%H%M')}.png")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        return img, url
    except Exception as e:
        print(f"  Could not download radar for {dt_rounded}: {e}")
        return None, url


def iem_image_to_refl(img, wrf_lat, wrf_lon):
    """
    Convert IEM n0q PNG composite to dBZ values on the WRF grid.

    IEM n0q encoding: dBZ = (pixel_value * 0.5) - 32.5
    pixel_value = 0 → no echo (NaN)
    pixel_value = 1-255 → -32.0 to 95.0 dBZ in 0.5 dBZ steps

    Returns dBZ array interpolated to WRF lat/lon grid.
    """
    if img is None:
        return None

    # IEM domain bounds (EPSG:4326)
    IEM_LON_MIN, IEM_LON_MAX = -126.0, -66.0
    IEM_LAT_MIN, IEM_LAT_MAX =   23.0,  50.0

    arr = np.array(img.convert('P'))
    img_h, img_w = arr.shape

    # Build pixel value -> dBZ lookup table
    dbz_lookup = np.full(256, np.nan)
    for pv in range(1, 256):
        dbz_lookup[pv] = (pv * 0.5) - 32.5

    # Apply lookup
    dbz = dbz_lookup[arr]

    # Mask below 0 dBZ (noise / clear air)
    dbz = np.where(dbz < 0, np.nan, dbz)

    # Map WRF grid points to IEM pixel coordinates
    lon_frac = (wrf_lon - IEM_LON_MIN) / (IEM_LON_MAX - IEM_LON_MIN)
    lat_frac = 1.0 - (wrf_lat - IEM_LAT_MIN) / (IEM_LAT_MAX - IEM_LAT_MIN)

    px = np.clip((lon_frac * img_w).astype(int), 0, img_w - 1)
    py = np.clip((lat_frac * img_h).astype(int), 0, img_h - 1)

    return dbz[py, px]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def add_state_borders(ax, lons, lats):
    """Add simple state border approximations using cartopy if available."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # If cartopy is available, add features
        ax.add_feature(cfeature.STATES.with_scale('50m'),
                       linewidth=0.6, edgecolor='black', facecolor='none')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=0.6, edgecolor='black')
    except Exception:
        # Cartopy not available — just add grid
        ax.grid(True, color='gray', lw=0.3, alpha=0.5)


def plot_refl_panel(ax, refl, lat, lon, title, stations=None,
                    cp_arrivals=None, vmin=0, vmax=75):
    """
    Plot a single reflectivity panel using pcolormesh for consistent,
    non-cartoonish appearance across WRF and observed panels.
    """
    refl_masked = np.where(np.isnan(refl) | (refl < 0), np.nan, refl)

    cf = ax.pcolormesh(lon, lat, refl_masked,
                       cmap=NWS_CMAP,
                       norm=REFL_NORM,
                       shading='auto')

    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)
    ax.grid(True, color='gray', lw=0.3, alpha=0.4)

    # Station markers
    if stations:
        for stn, (slat, slon) in stations.items():
            if (lon.min() <= slon <= lon.max() and
                    lat.min() <= slat <= lat.max()):
                ax.plot(slon, slat, 'k^', ms=5, zorder=6)
                ax.annotate(stn, (slon, slat), xytext=(3, 3),
                            textcoords='offset points',
                            fontsize=7, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', alpha=0.7))

    return cf


# ---------------------------------------------------------------------------
# Figure 1: Side-by-side at peak time
# ---------------------------------------------------------------------------

def plot_peak_comparison(wrf_refl, wrf_lat, wrf_lon, obs_refl,
                         peak_dt, output_path):
    """
    Side-by-side WRF simulated vs observed reflectivity at peak event time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             subplot_kw=dict())

    # WRF panel
    cf1 = plot_refl_panel(
        axes[0], wrf_refl, wrf_lat, wrf_lon,
        f'WRF Simulated Composite Reflectivity\n{peak_dt.strftime("%Y-%m-%d %H:%MZ")}',
        stations=HOUSTON_STATIONS)

    # Observed panel
    if obs_refl is not None:
        cf2 = plot_refl_panel(
            axes[1], obs_refl, wrf_lat, wrf_lon,
            f'NEXRAD Observed Composite Reflectivity\n{peak_dt.strftime("%Y-%m-%d %H:%MZ")}',
            stations=HOUSTON_STATIONS)
    else:
        axes[1].text(0.5, 0.5, 'Observed radar\nnot available',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=14)
        axes[1].set_title(f'NEXRAD Observed\n{peak_dt.strftime("%Y-%m-%d %H:%MZ")}',
                          fontsize=11, fontweight='bold')

    # Shared colorbar
    plt.colorbar(cf1, ax=axes, label='Composite Reflectivity (dBZ)',
                 orientation='horizontal', pad=0.05, shrink=0.6)

    fig.suptitle('WRF vs Observed Radar — Houston Derecho 16 May 2024',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Animated GIFs
# ---------------------------------------------------------------------------

def make_refl_gif(wrf_times_data, output_path, fps=3, mode='wrf',
                  wrf_lat=None, wrf_lon=None, obs_frames=None):
    """
    Create animated GIF of reflectivity evolution.

    mode = 'wrf'        : WRF only
    mode = 'obs'        : observed only
    mode = 'comparison' : side-by-side WRF vs obs
    """
    n_frames = len(wrf_times_data)
    if n_frames == 0:
        print("No frames to animate")
        return

    if mode == 'comparison':
        figsize = (18, 7)
        ncols = 2
    else:
        figsize = (10, 8)
        ncols = 1

    frames_data = []

    for idx, (dt, wrf_refl) in enumerate(wrf_times_data):
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1:
            axes = [axes]

        # WRF panel
        refl_masked = np.where(np.isnan(wrf_refl), np.nan, wrf_refl)
        cf = axes[0].pcolormesh(wrf_lon, wrf_lat, refl_masked,
                                cmap=NWS_CMAP, norm=REFL_NORM,
                                shading='auto')
        axes[0].set_xlim(wrf_lon.min(), wrf_lon.max())
        axes[0].set_ylim(wrf_lat.min(), wrf_lat.max())
        axes[0].set_title(f'WRF Simulated Reflectivity\n{dt.strftime("%Y-%m-%d %H:%MZ")}',
                          fontsize=11, fontweight='bold')
        axes[0].grid(True, color='gray', lw=0.3, alpha=0.4)

        for stn, (slat, slon) in HOUSTON_STATIONS.items():
            if wrf_lon.min() <= slon <= wrf_lon.max():
                axes[0].plot(slon, slat, 'k^', ms=5, zorder=6)
                axes[0].annotate(stn, (slon, slat), xytext=(2, 2),
                                 textcoords='offset points', fontsize=7,
                                 fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.15',
                                           fc='white', alpha=0.6))

        # Observed panel (comparison mode)
        if mode == 'comparison' and obs_frames is not None:
            obs = obs_frames[idx] if idx < len(obs_frames) else None
            if obs is not None:
                obs_masked = np.where(np.isnan(obs), np.nan, obs)
                axes[1].pcolormesh(wrf_lon, wrf_lat, obs_masked,
                                   cmap=NWS_CMAP, norm=REFL_NORM,
                                   shading='auto')
                axes[1].set_xlim(wrf_lon.min(), wrf_lon.max())
                axes[1].set_ylim(wrf_lat.min(), wrf_lat.max())
            else:
                axes[1].text(0.5, 0.5, 'No obs data',
                             ha='center', va='center',
                             transform=axes[1].transAxes)
            axes[1].set_title(f'NEXRAD Observed\n{dt.strftime("%Y-%m-%d %H:%MZ")}',
                              fontsize=11, fontweight='bold')
            axes[1].grid(True, color='gray', lw=0.3, alpha=0.4)

        plt.colorbar(cf, ax=axes, label='Reflectivity (dBZ)',
                     orientation='horizontal', pad=0.05, shrink=0.5)

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames_data.append(frame)
        plt.close()

        if (idx + 1) % 5 == 0:
            print(f"  Rendered {idx+1}/{n_frames} frames...")

    # Write GIF
    print(f"Writing GIF ({n_frames} frames)...")
    pil_frames = [Image.fromarray(f) for f in frames_data]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='WRF reflectivity vs observed radar comparison')
    parser.add_argument('--wrfout', nargs='+', required=True)
    parser.add_argument('--output-dir', default='figures/radar')
    parser.add_argument('--peak-time', required=True,
                        help='UTC peak time: "YYYY-MM-DD HH:MM"')
    parser.add_argument('--event-window-start', required=True)
    parser.add_argument('--event-window-end', required=True)
    parser.add_argument('--gif-fps', type=int, default=3)
    parser.add_argument('--skip-obs', action='store_true',
                        help='Skip observed radar download (offline mode)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peak_time          = pd.to_datetime(args.peak_time)
    event_start        = pd.to_datetime(args.event_window_start)
    event_end          = pd.to_datetime(args.event_window_end)

    # Load WRF time inventory
    print("Scanning WRF output files...")
    all_times = load_wrf_times(args.wrfout)
    event_times = [(f, t, dt) for f, t, dt in all_times
                   if event_start <= dt <= event_end]
    print(f"Found {len(event_times)} WRF time steps in event window")

    if not event_times:
        raise ValueError("No WRF time steps found in event window")

    # Extract all WRF reflectivity frames
    print("Extracting WRF reflectivity...")
    wrf_frames = []
    wrf_lat = None
    wrf_lon = None

    for fpath, tidx, dt in event_times:
        refl, lat, lon = extract_wrf_refl(fpath, tidx)
        if wrf_lat is None:
            wrf_lat = lat
            wrf_lon = lon
        wrf_frames.append((dt, refl))
        if len(wrf_frames) % 5 == 0:
            print(f"  Loaded {len(wrf_frames)}/{len(event_times)} frames")

    # Download observed radar frames
    obs_frames = []
    if not args.skip_obs:
        print("\nDownloading IEM NEXRAD composites...")
        for dt, _ in wrf_frames:
            img, url = download_iem_radar(dt)
            if img is not None:
                obs_refl = iem_image_to_refl(img, wrf_lat, wrf_lon)
                obs_frames.append(obs_refl)
                print(f"  Downloaded: {dt.strftime('%H:%MZ')}")
            else:
                obs_frames.append(None)
    else:
        print("Skipping observed radar download (--skip-obs)")
        obs_frames = [None] * len(wrf_frames)

    # --- Figure 1: Peak time side-by-side ---
    print("\nGenerating peak time comparison figure...")
    peak_idx = np.argmin([abs((dt - peak_time).total_seconds())
                          for dt, _ in wrf_frames])
    peak_dt, peak_wrf_refl = wrf_frames[peak_idx]
    peak_obs = obs_frames[peak_idx] if obs_frames else None

    plot_peak_comparison(
        peak_wrf_refl, wrf_lat, wrf_lon, peak_obs, peak_dt,
        output_dir / f'refl_comparison_{peak_dt.strftime("%Y%m%d_%H%MZ")}.png'
    )

    # --- Figure 2: WRF animation ---
    print("\nGenerating WRF reflectivity GIF...")
    make_refl_gif(
        wrf_frames,
        output_dir / 'wrf_refl_animation.gif',
        fps=args.gif_fps,
        mode='wrf',
        wrf_lat=wrf_lat,
        wrf_lon=wrf_lon
    )

    # --- Figure 3: Side-by-side comparison GIF ---
    if any(o is not None for o in obs_frames):
        print("\nGenerating WRF vs observed comparison GIF...")
        make_refl_gif(
            wrf_frames,
            output_dir / 'refl_comparison_animation.gif',
            fps=args.gif_fps,
            mode='comparison',
            wrf_lat=wrf_lat,
            wrf_lon=wrf_lon,
            obs_frames=obs_frames
        )

    print(f"\nAll figures saved to {output_dir}/")
    print("\nQuick MCS checklist — look for these in your WRF reflectivity:")
    print("  [ ] Organized convective line with >50 dBZ cores")
    print("  [ ] Stratiform precipitation region trailing the convective line")
    print("  [ ] Bow echo structure (forward-bulging line segment)")
    print("  [ ] Storm motion roughly west-to-east across Houston metro")
    print("  [ ] Timing: convective line over Houston area around 22-00Z")


if __name__ == '__main__':
    main()
