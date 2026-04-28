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

Produces an interactive HTML file you can rotate/zoom in any browser,
plus a static PNG for publications.

Example use:
    python coldpool_3d.py \
        --wrfout /data/scratch/a/procell2/messin_around/wrfout_d01_* \
        --time "2024-05-16 23:00" \
        --baseline-end "2024-05-16 12:00" \
        --output figures/coldpool_3d.html

Requirements:
    pip install plotly kaleido --break-system-packages
"""

import argparse
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Physical constants
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

# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

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
    print(f"Target time: {target_dt} (requested: {target_dt_req})")

    # Load WRF grid
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

    print(f"Domain subset: {j_max-j_min} x {i_max-i_min} grid points")

    # Compute temperature baseline from pre-storm times
    baseline_times = [(f, t, dt) for f, t, dt in all_times
                      if dt <= baseline_end_dt]
    print(f"Computing baseline from {len(baseline_times)} time steps...")

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

    # Extract target time fields
    print(f"Extracting 3D fields at {target_dt}...")
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


# ---------------------------------------------------------------------------
# 3D visualization
# ---------------------------------------------------------------------------

def make_3d_coldpool(data, output_html, output_png=None, subsample=3):
    """
    Create interactive 3D cold pool visualization using Plotly.

    Shows:
      - Blue isosurface: cold pool extent (T anomaly = -2K)
      - Deeper blue isosurface: cold pool core (T anomaly = -4K)
      - Wind speed volume slice: colored by wind speed
      - Surface T2 anomaly: colored tile at ground level
      - ASOS station markers
      - Houston reference marker
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("Plotly not installed. Run: pip install plotly kaleido --break-system-packages")
        return

    lat     = data['lat']
    lon     = data['lon']
    HGT     = data['HGT_AGL']
    T_anom  = data['T_anom']
    WSPD    = data['WSPD']
    T2_anom = data['T2_anom']
    WSPD_sfc= data['WSPD_sfc']
    target_dt = data['target_dt']
    max_h   = data['max_height']

    nz = T_anom.shape[0]

    # Subsample for performance
    ss = subsample
    lat_s  = lat[::ss, ::ss]
    lon_s  = lon[::ss, ::ss]
    HGT_s  = HGT[:, ::ss, ::ss]
    T_s    = T_anom[:, ::ss, ::ss]
    W_s    = WSPD[:, ::ss, ::ss]
    T2_s   = T2_anom[::ss, ::ss]
    Ws_s   = WSPD_sfc[::ss, ::ss]

    # Mask above max height
    height_ok = HGT_s <= max_h

    # Flatten for Plotly (needs 1D arrays)
    lon_flat = lon_s[np.newaxis, :, :].repeat(nz, axis=0)[height_ok].flatten()
    lat_flat = lat_s[np.newaxis, :, :].repeat(nz, axis=0)[height_ok].flatten()
    hgt_flat = HGT_s[height_ok].flatten()
    T_flat   = T_s[height_ok].flatten()
    W_flat   = W_s[height_ok].flatten()

    fig = go.Figure()

    # ---- 1. Wind speed volume (scatter3d colored by wind speed) ------------
    # Only show where wind speed > 10 m/s to avoid clutter
    wind_mask = W_flat > 10
    if wind_mask.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=lon_flat[wind_mask],
            y=lat_flat[wind_mask],
            z=hgt_flat[wind_mask],
            mode='markers',
            marker=dict(
                size=2,
                color=W_flat[wind_mask],
                colorscale='YlOrRd',
                cmin=10,
                cmax=25,
                opacity=0.15,
                colorbar=dict(
                    title='Wind Speed (m/s)',
                    x=1.05,
                    len=0.5,
                    y=0.75,
                )
            ),
            name='Wind Speed >10 m/s',
            showlegend=True,
        ))

    # ---- 2. Cold pool outer boundary (-2K isosurface) ----------------------
    # Use volume rendering for the cold pool
    cp_mask = (T_flat < -2) & (hgt_flat < max_h)
    if cp_mask.sum() > 100:
        fig.add_trace(go.Scatter3d(
            x=lon_flat[cp_mask],
            y=lat_flat[cp_mask],
            z=hgt_flat[cp_mask],
            mode='markers',
            marker=dict(
                size=3,
                color=T_flat[cp_mask],
                colorscale=[
                    [0.0, '#08306B'],   # deep navy (coldest)
                    [0.3, '#2171B5'],   # blue
                    [0.6, '#6BAED6'],   # light blue
                    [1.0, '#C6DBEF'],   # very light blue (-2K boundary)
                ],
                cmin=T_flat[cp_mask].min(),
                cmax=-2,
                opacity=0.4,
                colorbar=dict(
                    title='T Anomaly (K)',
                    x=1.05,
                    len=0.5,
                    y=0.25,
                )
            ),
            name='Cold Pool (T < −2K)',
            showlegend=True,
        ))

    # ---- 3. Cold pool core (-5K isosurface) --------------------------------
    core_mask = (T_flat < -5) & (hgt_flat < max_h)
    if core_mask.sum() > 50:
        fig.add_trace(go.Scatter3d(
            x=lon_flat[core_mask],
            y=lat_flat[core_mask],
            z=hgt_flat[core_mask],
            mode='markers',
            marker=dict(
                size=4,
                color='#08306B',
                opacity=0.7,
            ),
            name='Cold Pool Core (T < −5K)',
            showlegend=True,
        ))

    # ---- 4. Surface T2 anomaly (ground-level colored tiles) ----------------
    lon_sfc = lon_s.flatten()
    lat_sfc = lat_s.flatten()
    t2_sfc  = T2_s.flatten()
    ws_sfc  = Ws_s.flatten()

    # Surface cold pool footprint
    sfc_cp = t2_sfc < -2
    if sfc_cp.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=lon_sfc[sfc_cp],
            y=lat_sfc[sfc_cp],
            z=np.zeros(sfc_cp.sum()),
            mode='markers',
            marker=dict(
                size=4,
                color=t2_sfc[sfc_cp],
                colorscale='Blues_r',
                cmin=-8,
                cmax=0,
                opacity=0.8,
                symbol='circle',
            ),
            name='Surface Cold Pool (T2 < −2K)',
            showlegend=True,
        ))

    # Surface strong winds
    sfc_wind = ws_sfc > 10
    if sfc_wind.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=lon_sfc[sfc_wind],
            y=lat_sfc[sfc_wind],
            z=np.zeros(sfc_wind.sum()) + 50,  # slightly above ground
            mode='markers',
            marker=dict(
                size=3,
                color=ws_sfc[sfc_wind],
                colorscale='Reds',
                cmin=10,
                cmax=25,
                opacity=0.6,
                symbol='circle',
            ),
            name='Surface Wind >10 m/s',
            showlegend=True,
        ))

    # ---- 5. ASOS station markers -------------------------------------------
    for stn, (slat, slon) in HOUSTON_STATIONS.items():
        if (lon_s.min() <= slon <= lon_s.max() and
                lat_s.min() <= slat <= lat_s.max()):
            fig.add_trace(go.Scatter3d(
                x=[slon], y=[slat], z=[0],
                mode='markers+text',
                marker=dict(size=8, color='black', symbol='diamond'),
                text=[stn],
                textposition='top center',
                textfont=dict(size=10, color='black'),
                name=stn,
                showlegend=False,
            ))

    # ---- 6. Downtown Houston marker ----------------------------------------
    fig.add_trace(go.Scatter3d(
        x=[-95.35], y=[29.75], z=[0],
        mode='markers+text',
        marker=dict(size=12, color='gold', symbol='diamond',
                    line=dict(color='black', width=2)),
        text=['Downtown Houston'],
        textposition='top center',
        textfont=dict(size=11, color='darkred', family='Arial Black'),
        name='Downtown Houston',
        showlegend=True,
    ))

    # ---- 7. Vertical reference lines at Houston ----------------------------
    # Thin vertical line up to 5000m at Houston location
    fig.add_trace(go.Scatter3d(
        x=[-95.35, -95.35],
        y=[29.75, 29.75],
        z=[0, max_h],
        mode='lines',
        line=dict(color='gold', width=2, dash='dash'),
        name='Houston vertical ref',
        showlegend=False,
    ))

    # ---- Layout ------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(f'WRF Cold Pool 3D Structure — Houston Derecho<br>'
                  f'{target_dt.strftime("%Y-%m-%d %H:%MZ")}  |  '
                  f'Blue = cold pool (T anomaly < −2K)  |  '
                  f'Red/orange = wind speed > 10 m/s'),
            font=dict(size=14),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title='Longitude (°W)',
                tickformat='.1f',
                range=[lon_s.min(), lon_s.max()],
            ),
            yaxis=dict(
                title='Latitude (°N)',
                tickformat='.1f',
                range=[lat_s.min(), lat_s.max()],
            ),
            zaxis=dict(
                title='Height AGL (m)',
                range=[0, max_h],
            ),
            # Vertical exaggeration — stretch height to make structure visible
            aspectratio=dict(x=2, y=1.5, z=0.5),
            camera=dict(
                eye=dict(x=-1.8, y=-1.8, z=0.8),  # default viewing angle
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor='rgb(240, 245, 255)',
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10),
        ),
        width=1200,
        height=800,
        paper_bgcolor='white',
        annotations=[
            dict(
                text=(
                    'Rotate: left-click + drag  |  '
                    'Zoom: scroll  |  '
                    'Pan: right-click + drag  |  '
                    'Reset: double-click'
                ),
                xref='paper', yref='paper',
                x=0.5, y=-0.02,
                showarrow=False,
                font=dict(size=10, color='gray'),
                xanchor='center',
            )
        ]
    )

    # ---- Save HTML (interactive) -------------------------------------------
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    print(f"Interactive HTML saved: {output_html}")
    print("  Open in any browser to rotate/zoom the 3D figure")

    # ---- Save PNG (static) -------------------------------------------------
    if output_png:
        output_png = Path(output_png)
        try:
            fig.write_image(str(output_png), scale=2)
            print(f"Static PNG saved: {output_png}")
        except Exception as e:
            print(f"  PNG export failed (install kaleido): {e}")
            print("  Run: pip install kaleido --break-system-packages")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='3D cold pool visualization using Plotly')
    parser.add_argument('--wrfout',       nargs='+', required=True)
    parser.add_argument('--time',         default='2024-05-16 23:00',
                        help='Target time UTC: "YYYY-MM-DD HH:MM"')
    parser.add_argument('--baseline-end', default='2024-05-16 12:00',
                        help='End of pre-storm baseline period')
    parser.add_argument('--lat-min',      type=float, default=28.0)
    parser.add_argument('--lat-max',      type=float, default=34.0)
    parser.add_argument('--lon-min',      type=float, default=-99.0)
    parser.add_argument('--lon-max',      type=float, default=-93.0)
    parser.add_argument('--max-height',   type=int,   default=5000)
    parser.add_argument('--subsample',    type=int,   default=3,
                        help='Subsample factor for performance (default 3)')
    parser.add_argument('--output',       default='figures/coldpool_3d.html')
    parser.add_argument('--output-png',   default='figures/coldpool_3d.png')
    args = parser.parse_args()

    print("Extracting WRF 3D fields...")
    data = extract_3d_fields(
        args.wrfout,
        target_time  = args.time,
        baseline_end = args.baseline_end,
        lat_bounds   = (args.lat_min, args.lat_max),
        lon_bounds   = (args.lon_min, args.lon_max),
        max_height_m = args.max_height,
    )

    print("\nBuilding 3D visualization...")
    make_3d_coldpool(
        data,
        output_html = args.output,
        output_png  = args.output_png,
        subsample   = args.subsample,
    )


if __name__ == '__main__':
    main()
