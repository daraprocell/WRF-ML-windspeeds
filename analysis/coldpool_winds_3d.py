#!/usr/bin/env python3
"""
coldpool_3d.py
--------------
3D visualization of WRF cold pool structure using Plotly.

Shows the cold pool as a 3D temperature anomaly isosurface, with:
  - Blue isosurface: cold pool extent (T anomaly < -2K)
  - Wind speed volume: colored by horizontal wind magnitude
  - ASOS station markers at surface
  - Houston reference point

Produces an interactive HTML file.

Example use:
    python coldpool_3d.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d01_* \
        --time "2024-05-16 23:00" \
        --baseline-end "2024-05-16 12:00" \
        --output figures/coldpool_3d.html
"""

import argparse
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
import plotly.graph_objects as go
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import warnings
warnings.filterwarnings('ignore')

R_cp = 0.2854
P0   = 100000.0

# ASOS stations
HOUSTON_STATIONS = {
    'KHOU': (29.64, -95.28),
    'KIAH': (29.98, -95.36),
    'KDWH': (30.07, -95.56),
    'KSGR': (29.62, -95.66),
    'KCLL': (30.59, -96.36),
    'KTME': (29.81, -95.90),
}


def load_wrf_times(wrfout_files):
    entries = []
    for fpath in sorted(wrfout_files):
        with Dataset(fpath, 'r') as nc:
            times_raw = nc.variables['Times'][:]
            for t in range(times_raw.shape[0]):
                tstr = ''.join([c.decode() for c in times_raw[t]]).replace('_', ' ')
                dt = pd.to_datetime(tstr)
                entries.append((fpath, t, dt))
    return sorted(entries, key=lambda x: x[2])


def extract_3d_fields(wrfout_files, target_time, baseline_end,
                      lat_bounds, lon_bounds, max_height_m=5000):
    """
    Extract 3D temperature anomaly and wind speed on a regular height grid
    within the specified lat/lon bounds.

    Returns dict with:
        lat, lon     : 1D arrays of unique lats/lons in domain
        height       : 1D array of height levels (m AGL)
        T_anom       : 3D array (nz, ny, nx) temperature anomaly (K)
        WSPD         : 3D array (nz, ny, nx) horizontal wind speed (m/s)
        U10, V10     : 2D arrays (ny, nx) surface wind
        T2_anom      : 2D array (ny, nx) 2m temperature anomaly
        target_dt    : actual datetime used
    """
    all_times = load_wrf_times(wrfout_files)
    baseline_end_dt = pd.to_datetime(baseline_end)
    target_dt_req   = pd.to_datetime(target_time)

    # Find closest time step to target
    diffs      = [abs((dt - target_dt_req).total_seconds()) for _, _, dt in all_times]
    target_idx = int(np.argmin(diffs))
    target_fpath, target_tidx, target_dt = all_times[target_idx]

    with Dataset(target_fpath, 'r') as nc:
        wrf_lat = nc.variables['XLAT'][0, :, :]
        wrf_lon = nc.variables['XLONG'][0, :, :]

    # Find grid indices within lat/lon bounds
    lat_mask = (wrf_lat >= lat_bounds[0]) & (wrf_lat <= lat_bounds[1])
    lon_mask = (wrf_lon >= lon_bounds[0]) & (wrf_lon <= lon_bounds[1])
    domain_mask = lat_mask & lon_mask

    rows = np.where(domain_mask.any(axis=1))[0]
    cols = np.where(domain_mask.any(axis=0))[0]
    j_min, j_max = rows[0], rows[-1] + 1
    i_min, i_max = cols[0], cols[-1] + 1

    sub_lat = wrf_lat[j_min:j_max, i_min:i_max]
    sub_lon = wrf_lon[j_min:j_max, i_min:i_max]

    baseline_times = [(f, t, dt) for f, t, dt in all_times
                      if dt <= baseline_end_dt]

    T_baseline_sum = None
    T2_baseline_sum = None
    n_baseline = 0

    for fpath, tidx, dt in baseline_times:
        with Dataset(fpath, 'r') as nc:
            T_pert = nc.variables['T'][tidx, :, j_min:j_max, i_min:i_max]
            P_pert = nc.variables['P'][tidx, :, j_min:j_max, i_min:i_max]
            PB     = nc.variables['PB'][tidx, :, j_min:j_max, i_min:i_max]
            T2     = nc.variables['T2'][tidx, j_min:j_max, i_min:i_max]

        theta   = T_pert + 300.0
        P_total = P_pert + PB
        T_K     = theta * (P_total / P0) ** R_cp

        if T_baseline_sum is None:
            T_baseline_sum  = T_K.copy()
            T2_baseline_sum = T2.copy().astype(float)
        else:
            T_baseline_sum  += T_K
            T2_baseline_sum += T2
        n_baseline += 1

    T_baseline  = T_baseline_sum / n_baseline
    T2_baseline = T2_baseline_sum / n_baseline

    with Dataset(target_fpath, 'r') as nc:
        T_pert = nc.variables['T'][target_tidx, :, j_min:j_max, i_min:i_max]
        P_pert = nc.variables['P'][target_tidx, :, j_min:j_max, i_min:i_max]
        PB     = nc.variables['PB'][target_tidx, :, j_min:j_max, i_min:i_max]
        PH     = nc.variables['PH'][target_tidx, :, j_min:j_max, i_min:i_max]
        PHB    = nc.variables['PHB'][target_tidx, :, j_min:j_max, i_min:i_max]

        # U wind — staggered west-east, take subset carefully
        U_stag = nc.variables['U'][target_tidx, :, j_min:j_max, i_min:i_max+1]
        U      = 0.5 * (U_stag[:, :, :-1] + U_stag[:, :, 1:])

        # V wind — staggered south-north
        V_stag = nc.variables['V'][target_tidx, :, j_min:j_max+1, i_min:i_max]
        V      = 0.5 * (V_stag[:, :-1, :] + V_stag[:, 1:, :])

        U10 = nc.variables['U10'][target_tidx, j_min:j_max, i_min:i_max]
        V10 = nc.variables['V10'][target_tidx, j_min:j_max, i_min:i_max]
        T2  = nc.variables['T2'][target_tidx, j_min:j_max, i_min:i_max]

    # Compute actual temperature and anomaly
    theta   = T_pert + 300.0
    P_total = P_pert + PB
    T_K     = theta * (P_total / P0) ** R_cp
    T_anom  = T_K - T_baseline

    # Geopotential height AGL
    HGT_stag = (PH + PHB) / 9.81
    HGT      = 0.5 * (HGT_stag[:-1, :, :] + HGT_stag[1:, :, :])
    terrain  = HGT[0, :, :]
    HGT_AGL  = HGT - terrain[np.newaxis, :, :]

    # Horizontal wind speed
    WSPD = np.sqrt(U**2 + V**2)

    # Mask above max height
    height_mask = HGT_AGL <= max_height_m

    # T2 anomaly
    T2_anom = T2 - T2_baseline

    # Surface wind speed
    WSPD_sfc = np.sqrt(U10**2 + V10**2)

    print(f"  Cold pool extent (<-2K): "
          f"{(T_anom[height_mask] < -2).sum()} grid points")
    print(f"  Max wind speed in domain: {WSPD[height_mask].max():.1f} m/s")
    print(f"  Min T anomaly: {T_anom[height_mask].min():.1f} K")

    return {
        'lat':       sub_lat,
        'lon':       sub_lon,
        'HGT_AGL':   HGT_AGL,
        'T_anom':    T_anom,
        'WSPD':      WSPD,
        'WSPD_sfc':  WSPD_sfc,
        'T2_anom':   T2_anom,
        'target_dt': target_dt,
        'max_height': max_height_m,
    }



def _add_markers(fig, lon_s, lat_s):
    """Add ASOS stations and Houston marker to a Plotly figure."""

    for stn, (slat, slon) in HOUSTON_STATIONS.items():
        if (lon_s.min() <= slon <= lon_s.max() and
                lat_s.min() <= slat <= lat_s.max()):
            fig.add_trace(go.Scatter3d(
                x=[slon], y=[slat], z=[0],
                mode='markers+text',
                marker=dict(size=5, color='black', symbol='diamond'),
                text=[stn],
                textposition='top center',
                textfont=dict(size=9, color='black'),
                name=stn,
                showlegend=False,
            ))
    
    coast = cfeature.COASTLINE.with_scale('10m')
    lon_min, lon_max = float(lon_s.min()), float(lon_s.max())
    lat_min, lat_max = float(lat_s.min()), float(lat_s.max())
    clip_box = sgeom.box(lon_min, lat_min, lon_max, lat_max)

    coast_lons, coast_lats = [], []
    for geom in coast.geometries():
        clipped = geom.intersection(clip_box)
        if clipped.is_empty:
            continue
        segs = [clipped] if clipped.geom_type == 'LineString' \
                else list(clipped.geoms)
        for seg in segs:
            xs, ys = seg.xy
            coast_lons.extend(list(xs) + [None])
            coast_lats.extend(list(ys) + [None])

    if coast_lons:
        fig.add_trace(go.Scatter3d(
            x=coast_lons, y=coast_lats,
            z=[0 if v is not None else None for v in coast_lons],
            mode='lines',
            line=dict(color='black', width=2),
            name='Coastline',
            showlegend=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=[-95.35], y=[29.75], z=[0],
        mode='markers+text',
        marker=dict(size=6, color='gold',
                    symbol='diamond',
                    line=dict(color='black', width=2)),
        text=['Downtown Houston'],
        textposition='top center',
        textfont=dict(size=11, color='darkred', family='Arial Black'),
        name='Downtown Houston',
        showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[-95.35, -95.35], y=[29.75, 29.75], z=[0, 5000],
        mode='lines',
        line=dict(color='gold', width=2, dash='dash'),
        showlegend=False,
    ))


def _scene_layout(lon_s, lat_s, max_h):
    return dict(
        xaxis=dict(title='Longitude (°W)', tickformat='.1f',
                   range=[lon_s.min(), lon_s.max()]),
        yaxis=dict(title='Latitude (°N)',  tickformat='.1f',
                   range=[lat_s.min(), lat_s.max()]),
        zaxis=dict(title='Height AGL (m)', range=[0, max_h]),
        aspectratio=dict(x=2, y=1.5, z=0.5),
        camera=dict(eye=dict(x=-1.8, y=-1.8, z=0.8),
                    up=dict(x=0, y=0, z=1)),
        bgcolor='rgb(240, 245, 255)',
    )


def _legend_style():
    return dict(x=0.01, y=0.99,
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='gray', borderwidth=1,
                font=dict(size=10))


def _controls_annotation():
    return dict(
        text='Rotate: left-click + drag  |  Zoom: scroll  |  '
             'Pan: right-click + drag  |  Reset: double-click',
        xref='paper', yref='paper', x=0.5, y=-0.02,
        showarrow=False, font=dict(size=10, color='gray'),
        xanchor='center',
    )


def _save_figure(fig, output_html, output_png):
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))



# Figure 1: Temperature anomaly 

def make_3d_temperature(data, output_html, output_png=None, subsample=3):
    """3D cold pool temperature anomaly — single continuous blue gradient."""

    lat, lon = data['lat'], data['lon']
    HGT, T_anom = data['HGT_AGL'], data['T_anom']
    target_dt = data['target_dt']
    max_h, nz = data['max_height'], T_anom.shape[0]

    ss = subsample
    lat_s = lat[::ss, ::ss];  lon_s = lon[::ss, ::ss]
    HGT_s = HGT[:, ::ss, ::ss];  T_s = T_anom[:, ::ss, ::ss]

    ok = HGT_s <= max_h
    lon_f = lon_s[np.newaxis].repeat(nz, 0)[ok].flatten()
    lat_f = lat_s[np.newaxis].repeat(nz, 0)[ok].flatten()
    hgt_f = HGT_s[ok].flatten()
    T_f   = T_s[ok].flatten()

    fig = go.Figure()

    cp = T_f < -2
    if cp.sum() > 100:
        t_min = float(np.percentile(T_f[cp], 2)) 
        fig.add_trace(go.Scatter3d(
            x=lon_f[cp], y=lat_f[cp], z=hgt_f[cp],
            mode='markers',
            marker=dict(
                size=3,
                color=T_f[cp],
                colorscale='Blues',
                cmin=t_min,
                cmax=-2,
                opacity=0.4,
                colorbar=dict(
                    title=dict(text='T Anomaly (K)', font=dict(size=12)),
                    x=1.02, len=0.7, y=0.5,
                    tickfont=dict(size=11),
                ),
            ),
            name='Cold Pool (T < −2K)',
        ))

    _add_markers(fig, lon_s, lat_s)

    t_min_label = float(np.min(T_f[T_f < -2])) if (T_f < -2).any() else -2
    fig.update_layout(
        title=dict(
            text=(f'WRF Cold Pool Temperature Anomaly — Houston Derecho<br>'
                  f'{target_dt.strftime("%Y-%m-%d %H:%MZ")}'),
            font=dict(size=13), x=0.5,
        ),
        scene=_scene_layout(lon_s, lat_s, max_h),
        legend=_legend_style(),
        width=1100, height=750,
        paper_bgcolor='white',
        annotations=[_controls_annotation()],
    )
    _save_figure(fig, output_html, output_png)


# Figure 2: Wind speed

def make_3d_winds(data, output_html, output_png=None, subsample=3):
    """3D wind speed structure — single continuous warm gradient."""

    lat, lon = data['lat'], data['lon']
    HGT, WSPD = data['HGT_AGL'], data['WSPD']
    target_dt = data['target_dt']
    max_h, nz = data['max_height'], WSPD.shape[0]

    ss = subsample
    lat_s = lat[::ss, ::ss];  lon_s = lon[::ss, ::ss]
    HGT_s = HGT[:, ::ss, ::ss];  W_s = WSPD[:, ::ss, ::ss]

    ok = HGT_s <= max_h
    lon_f = lon_s[np.newaxis].repeat(nz, 0)[ok].flatten()
    lat_f = lat_s[np.newaxis].repeat(nz, 0)[ok].flatten()
    hgt_f = HGT_s[ok].flatten()
    W_f   = W_s[ok].flatten()

    fig = go.Figure()

    # Single trace — all points above threshold, one continuous gradient
    WIND_THRESH = 2.0
    wm = W_f > WIND_THRESH
    if wm.sum() > 0:
        w_max = float(np.percentile(W_f[wm], 99))
        fig.add_trace(go.Scatter3d(
            x=lon_f[wm], y=lat_f[wm], z=hgt_f[wm],
            mode='markers',
            marker=dict(
                size=3,
                color=W_f[wm],
                colorscale='YlOrRd',
                cmin=WIND_THRESH,
                cmax=w_max,
                opacity=0.35,
                colorbar=dict(
                    title=dict(text='Wind Speed (m/s)', font=dict(size=12)),
                    x=1.02, len=0.7, y=0.5,
                    tickfont=dict(size=11),
                ),
            ),
            name=f'Wind Speed >{WIND_THRESH:.0f} m/s',
        ))

    _add_markers(fig, lon_s, lat_s)

    w_max_label = float(np.max(W_f[W_f > WIND_THRESH])) \
                  if (W_f > WIND_THRESH).any() else WIND_THRESH
    fig.update_layout(
        title=dict(
            text=(f'WRF Wind Speed Structure — Houston Derecho<br>'
                  f'{target_dt.strftime("%Y-%m-%d %H:%MZ")}  |  '
                  f'{w_max_label:.0f} m/s'),
            font=dict(size=13), x=0.5,
        ),
        scene=_scene_layout(lon_s, lat_s, max_h),
        legend=_legend_style(),
        width=1100, height=750,
        paper_bgcolor='white',
        annotations=[_controls_annotation()],
    )
    _save_figure(fig, output_html, output_png)



def main():
    parser = argparse.ArgumentParser(
        description='3D cold pool visualization using Plotly — two separate figures')
    parser.add_argument('--wrfout',       nargs='+', required=True)
    parser.add_argument('--time',         default='2024-05-16 23:00')
    parser.add_argument('--baseline-end', default='2024-05-16 12:00')
    parser.add_argument('--lat-min',      type=float, default=28.0)
    parser.add_argument('--lat-max',      type=float, default=34.0)
    parser.add_argument('--lon-min',      type=float, default=-99.0)
    parser.add_argument('--lon-max',      type=float, default=-93.0)
    parser.add_argument('--max-height',   type=int,   default=5000)
    parser.add_argument('--subsample',    type=int,   default=3)
    parser.add_argument('--output-dir',   default='figures')
    args = parser.parse_args()

    data = extract_3d_fields(
        args.wrfout,
        target_time  = args.time,
        baseline_end = args.baseline_end,
        lat_bounds   = (args.lat_min, args.lat_max),
        lon_bounds   = (args.lon_min, args.lon_max),
        max_height_m = args.max_height,
    )

    out = Path(args.output_dir)

    make_3d_temperature(
        data,
        output_html = out / 'coldpool_3d_temperature.html',
        output_png  = out / 'coldpool_3d_temperature.png',
        subsample   = args.subsample,
    )

    make_3d_winds(
        data,
        output_html = out / 'coldpool_3d_winds.html',
        output_png  = out / 'coldpool_3d_winds.png',
        subsample   = args.subsample,
    )


if __name__ == '__main__':
    main()