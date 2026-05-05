#!/usr/bin/env python3
"""
radar.py
-----------------------
Compare WRF simulated reflectivity (REFL_10CM) against IEM NEXRAD composite
reflectivity for the Houston derecho event.

Produces:
  1. refl_comparison_{time}.png  - one side-by-side PNG per WRF time step
  2. wrf_refl.gif      - WRF reflectivity evolution
  3. obs_refl.gif      - observed reflectivity evolution
  4. refl_comparison.gif         - side-by-side WRF vs obs GIF

Use --peak-time to produce a single comparison PNG for one specific time
instead of looping over all WRF time steps.

Use --obs-time-offset MINUTES to compare WRF at each WRF time against
observed radar from MINUTES earlier. Positive values mean the simulated
storm lags the observed storm. When this flag is used, gifs are
skipped to analyze just the tiem offset.

Examples:
    # Normal simultaneous comparison (all WRF time steps + gifs)
    python radar.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d02_* \
        --output-dir figures/radar \
        --event-window-start "2024-05-16 18:00" \
        --event-window-end "2024-05-17 02:00"

    # Single-time comparison at one specific peak time
    python radar.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d02_* \
        --output-dir figures/radar \
        --event-window-start "2024-05-16 18:00" \
        --event-window-end "2024-05-17 02:00" \
        --peak-time "2024-05-17 01:00"

    # Lag-corrected single-time comparison: WRF at 0100Z vs NEXRAD 75 min earlier
    python radar.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d02_* \
        --output-dir figures/radar \
        --event-window-start "2024-05-16 18:00" \
        --event-window-end "2024-05-17 02:00" \
        --peak-time "2024-05-17 01:00" \
        --obs-time-offset 75
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
from io import BytesIO
from PIL import Image
from netCDF4 import Dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


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
    Extract column max simulated reflectivity from a WRF output file.
    Uses REFL_10CM (3D) and takes the column maximum (composite reflectivity).
    """
    with Dataset(fpath, 'r') as nc:
        refl_3d = nc.variables['REFL_10CM'][tidx, :, :, :]  # (nz, ny, nx)
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]

    # Column max = composite reflectivity
    refl_comp = np.max(refl_3d, axis=0)

    # Mask negative values (clear air)
    refl_comp = np.where(refl_comp < 0, np.nan, refl_comp)

    return refl_comp, wrf_lat, wrf_lon


def download_iem_radar(dt):
    """
    Download IEM NEXRAD composite reflectivity PNG for a given UTC datetime.
    Returns PIL Image or None if unavailable.

    IEM files are named n0q_YYYYMMDDHH\u041c\u041c.png at 5 minute intervals.
    We find the nearest 5 minute timestamp.
    """
    # Round to nearest 5 min
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
        print(f"Could not download radar for {dt_rounded}: {e}")
        return None, url


def iem_image_to_refl(img, wrf_lat, wrf_lon):
    """
    Convert IEM n0q PNG composite to dBZ values on the WRF grid.

    IEM n0q encoding: dBZ = (pixel_value * 0.5) - 32.5
    pixel_value = 0 \u2192 no echo (NaN)
    pixel_value = 1-255 \u2192 -32.0 to 95.0 dBZ in 0.5 dBZ steps

    Returns dBZ array interpolated to WRF lat/lon grid.
    """
    if img is None:
        return None

    # IEM domain bounds 
    IEM_LON_MIN, IEM_LON_MAX = -126.0, -66.0
    IEM_LAT_MIN, IEM_LAT_MAX =   23.0,  50.0

    arr = np.array(img.convert('P'))
    img_h, img_w = arr.shape

    # Build pixel value 
    dbz_lookup = np.full(256, np.nan)
    for pv in range(1, 256):
        dbz_lookup[pv] = (pv * 0.5) - 32.5

    dbz = dbz_lookup[arr]

    # Mask below 0 dBZ (noise / clear air)
    dbz = np.where(dbz < 0, np.nan, dbz)

    # Map WRF grid points to IEM pixel coordinates
    lon_frac = (wrf_lon - IEM_LON_MIN) / (IEM_LON_MAX - IEM_LON_MIN)
    lat_frac = 1.0 - (wrf_lat - IEM_LAT_MIN) / (IEM_LAT_MAX - IEM_LAT_MIN)

    px = np.clip((lon_frac * img_w).astype(int), 0, img_w - 1)
    py = np.clip((lat_frac * img_h).astype(int), 0, img_h - 1)

    return dbz[py, px]


def plot_refl_panel(ax, refl, lat, lon, title, stations=None,
                    cp_arrivals=None, vmin=0, vmax=75):
    """
    Plot a single reflectivity panel using pcolormesh for consistent
    appearance across WRF and observed panels.
    """
    refl_masked = np.where(np.isnan(refl) | (refl < 0), np.nan, refl)

    cf = ax.pcolormesh(lon, lat, refl_masked,
                       cmap=NWS_CMAP,
                       norm=REFL_NORM,
                       shading='auto',
                       transform=ccrs.PlateCarree())

    # Cartopy map features
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                   linewidth=1.2, edgecolor='black', zorder=5)
    ax.add_feature(cfeature.STATES.with_scale('10m'),
                   linewidth=0.7, edgecolor='#444444',
                   facecolor='none', zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                   linewidth=0.8, edgecolor='#444444', zorder=5)

    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                  crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.gridlines(draw_labels=True, linewidth=0.4,
                 color='gray', alpha=0.5, linestyle='--')

    # Station markers
    if stations:
        for stn, (slat, slon) in stations.items():
            if (lon.min() <= slon <= lon.max() and
                    lat.min() <= slat <= lat.max()):
                ax.plot(slon, slat, 'k^', ms=5, zorder=6,
                        transform=ccrs.PlateCarree())
                ax.annotate(stn, (slon, slat), xytext=(3, 3),
                            textcoords='offset points',
                            fontsize=7, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', alpha=0.7),
                            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax))

    return cf


# Figure 1: Side by side comparison

def plot_comparison(wrf_refl, wrf_lat, wrf_lon, obs_refl,
                    wrf_dt, output_path, obs_dt=None):
    """
    Side by side WRF simulated vs observed reflectivity at a single time.
    If obs_dt is provided and differs from wrf_dt, the observed panel is
    labeled with the observed-radar time and the time offset (in minutes)
    is shown in the figure title.
    """

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 0.04],
        hspace=0.18, wspace=0.05,
        left=0.05, right=0.97, top=0.82, bottom=0.05,
    )
    ax_wrf = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_obs = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_cb  = fig.add_subplot(gs[1, :])  # colorbar spans full width

    # WRF panel
    cf1 = plot_refl_panel(
        ax_wrf, wrf_refl, wrf_lat, wrf_lon,
        f'WRF Simulated Composite Reflectivity\n{wrf_dt.strftime("%Y-%m-%d %H:%MZ")}',
        stations=HOUSTON_STATIONS)

    # Observed panel
    obs_label_dt = obs_dt if obs_dt is not None else wrf_dt

    cf2 = plot_refl_panel(
        ax_obs, obs_refl, wrf_lat, wrf_lon,
        f'NEXRAD Observed Composite Reflectivity\n{obs_label_dt.strftime("%Y-%m-%d %H:%MZ")}',
        stations=HOUSTON_STATIONS)

    plt.colorbar(cf1, cax=ax_cb, orientation='horizontal',
                 label='Composite Reflectivity (dBZ)')

    if obs_dt is not None and obs_dt != wrf_dt:
        offset_min = (wrf_dt - obs_dt).total_seconds() / 60.0
        suptitle = (
            'WRF vs Observed Radar - Houston Derecho 16 May 2024\n'
            f'(WRF lagged by {offset_min:.0f} min relative to observations)'
        )
    else:
        suptitle = 'WRF vs Observed Radar - Houston Derecho 16 May 2024'

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.94)

    plt.savefig(output_path, dpi=200)
    plt.close()


# Figure 2: GIFs

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
        fig, axes = plt.subplots(1, ncols, figsize=figsize,
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        if ncols == 1:
            axes = [axes]

        # WRF panel
        refl_masked = np.where(np.isnan(wrf_refl), np.nan, wrf_refl)
        cf = axes[0].pcolormesh(wrf_lon, wrf_lat, refl_masked,
                                cmap=NWS_CMAP, norm=REFL_NORM,
                                shading='auto',
                                transform=ccrs.PlateCarree())
        axes[0].add_feature(cfeature.COASTLINE.with_scale('10m'),
                            linewidth=1.0, edgecolor='black', zorder=5)
        axes[0].add_feature(cfeature.STATES.with_scale('10m'),
                            linewidth=0.6, edgecolor='#444444',
                            facecolor='none', zorder=5)
        axes[0].add_feature(cfeature.BORDERS.with_scale('10m'),
                            linewidth=0.7, edgecolor='#444444', zorder=5)
        axes[0].set_extent([wrf_lon.min(), wrf_lon.max(),
                            wrf_lat.min(), wrf_lat.max()],
                           crs=ccrs.PlateCarree())
        axes[0].set_title(f'WRF Simulated Reflectivity\n{dt.strftime("%Y-%m-%d %H:%MZ")}',
                          fontsize=11, fontweight='bold')
        axes[0].gridlines(draw_labels=True, linewidth=0.3,
                          color='gray', alpha=0.5, linestyle='--')

        for stn, (slat, slon) in HOUSTON_STATIONS.items():
            if wrf_lon.min() <= slon <= wrf_lon.max():
                axes[0].plot(slon, slat, 'k^', ms=5, zorder=6,
                             transform=ccrs.PlateCarree())
                axes[0].annotate(stn, (slon, slat), xytext=(2, 2),
                                 textcoords='offset points', fontsize=7,
                                 fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.15',
                                           fc='white', alpha=0.6),
                                 xycoords=ccrs.PlateCarree()._as_mpl_transform(axes[0]))

        # Observed panel (comparison mode)
        if mode == 'comparison' and obs_frames is not None:
            obs = obs_frames[idx] if idx < len(obs_frames) else None
            if obs is not None:
                obs_masked = np.where(np.isnan(obs), np.nan, obs)
                axes[1].pcolormesh(wrf_lon, wrf_lat, obs_masked,
                                   cmap=NWS_CMAP, norm=REFL_NORM,
                                   shading='auto',
                                   transform=ccrs.PlateCarree())
                axes[1].add_feature(cfeature.COASTLINE.with_scale('10m'),
                                    linewidth=1.0, edgecolor='black', zorder=5)
                axes[1].add_feature(cfeature.STATES.with_scale('10m'),
                                    linewidth=0.6, edgecolor='#444444',
                                    facecolor='none', zorder=5)
                axes[1].add_feature(cfeature.BORDERS.with_scale('10m'),
                                    linewidth=0.7, edgecolor='#444444', zorder=5)
                axes[1].set_extent([wrf_lon.min(), wrf_lon.max(),
                                    wrf_lat.min(), wrf_lat.max()],
                                   crs=ccrs.PlateCarree())
            else:
                axes[1].text(0.5, 0.5, 'No obs data',
                             ha='center', va='center',
                             transform=axes[1].transAxes)
            axes[1].set_title(f'NEXRAD Observed\n{dt.strftime("%Y-%m-%d %H:%MZ")}',
                              fontsize=11, fontweight='bold')
            axes[1].gridlines(draw_labels=True, linewidth=0.3,
                              color='gray', alpha=0.5, linestyle='--')

        plt.colorbar(cf, ax=axes, label='Reflectivity (dBZ)',
                     orientation='horizontal', pad=0.05, shrink=0.5)

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames_data.append(frame)
        plt.close()

    pil_frames = [Image.fromarray(f) for f in frames_data]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )


def main():
    parser = argparse.ArgumentParser(
        description='WRF reflectivity vs observed radar comparison')
    parser.add_argument('--wrfout', nargs='+', required=True)
    parser.add_argument('--output-dir', default='figures/radar')
    parser.add_argument('--event-window-start', required=True,
                        help='Start of event window: "YYYY-MM-DD HH:MM" (UTC)')
    parser.add_argument('--event-window-end', required=True,
                        help='End of event window: "YYYY-MM-DD HH:MM" (UTC)')
    parser.add_argument('--peak-time', default=None,
                        help='Optional UTC time "YYYY-MM-DD HH:MM" for which '
                             'a single comparison PNG will be produced. '
                             'When provided, gifs are skipped and only '
                             'the one PNG closest to this time is generated.')
    parser.add_argument('--gif-fps', type=int, default=3)
    parser.add_argument('--skip-obs', action='store_true',
                        help='Skip observed radar download (offline mode)')
    parser.add_argument('--obs-time-offset', type=float, default=0.0,
                        help='Minutes to subtract from each WRF time when '
                             'sampling observed radar. Use a positive value '
                             'when WRF lags observations. Example: '
                             '--obs-time-offset 75 pairs WRF at each time '
                             'with observed radar from 75 min earlier. '
                             'When non-zero, gifs are skipped.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_start = pd.to_datetime(args.event_window_start)
    event_end   = pd.to_datetime(args.event_window_end)
    obs_offset_min = args.obs_time_offset

    # Load WRF time inventory
    all_times = load_wrf_times(args.wrfout)
    event_times = [(f, t, dt) for f, t, dt in all_times
                   if event_start <= dt <= event_end]
    if not event_times:
        raise ValueError("No WRF time steps found in event window")

    # Extract all WRF reflectivity frames
    wrf_frames = []
    wrf_lat = None
    wrf_lon = None

    for fpath, tidx, dt in event_times:
        refl, lat, lon = extract_wrf_refl(fpath, tidx)
        if wrf_lat is None:
            wrf_lat = lat
            wrf_lon = lon
        wrf_frames.append((dt, refl))

    # If --peak-time was provided, narrow to the single nearest WRF frame
    if args.peak_time is not None:
        peak_time = pd.to_datetime(args.peak_time)
        peak_idx = np.argmin([abs((dt - peak_time).total_seconds())
                              for dt, _ in wrf_frames])
        png_frames = [wrf_frames[peak_idx]]
    else:
        png_frames = wrf_frames

    # Side by side comparison PNG for each selected WRF time step
    for wrf_dt, wrf_refl in png_frames:
        obs_dt = wrf_dt - pd.Timedelta(minutes=obs_offset_min)

        obs_refl = None
        if not args.skip_obs:
            obs_img, _ = download_iem_radar(obs_dt)
            if obs_img is not None:
                obs_refl = iem_image_to_refl(obs_img, wrf_lat, wrf_lon)

        # Output filename reflects whether an offset was applied
        if obs_offset_min != 0.0:
            out_name = (f'refl_comparison_{wrf_dt.strftime("%Y%m%d_%H%MZ")}_'
                        f'vs_obs_{obs_dt.strftime("%H%MZ")}.png')
        else:
            out_name = f'refl_comparison_{wrf_dt.strftime("%Y%m%d_%H%MZ")}.png'

        plot_comparison(
            wrf_refl, wrf_lat, wrf_lon, obs_refl, wrf_dt,
            output_dir / out_name,
            obs_dt=obs_dt if obs_offset_min != 0.0 else None,
        )

        return

    # Download obs at every WRF frame time
    obs_frames = []
    if not args.skip_obs:
        for dt, _ in wrf_frames:
            img, url = download_iem_radar(dt)
            if img is not None:
                obs_refl = iem_image_to_refl(img, wrf_lat, wrf_lon)
                obs_frames.append(obs_refl)
            else:
                obs_frames.append(None)
    else:
        obs_frames = [None] * len(wrf_frames)

    # WRF only gif
    make_refl_gif(
        wrf_frames,
        output_dir / 'wrf_refl.gif',
        fps=args.gif_fps,
        mode='wrf',
        wrf_lat=wrf_lat,
        wrf_lon=wrf_lon
    )

    # Side by side comparison gif
    if any(o is not None for o in obs_frames):
        make_refl_gif(
            wrf_frames,
            output_dir / 'refl_comparison.gif',
            fps=args.gif_fps,
            mode='comparison',
            wrf_lat=wrf_lat,
            wrf_lon=wrf_lon,
            obs_frames=obs_frames
        )


if __name__ == '__main__':
    main()