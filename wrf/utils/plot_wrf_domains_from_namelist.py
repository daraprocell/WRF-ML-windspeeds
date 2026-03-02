#!/usr/bin/env python3
"""
plot_wrf_domains.py
-------------------
Parse a namelist.wps file and plot all WRF domains (outer + nested) on a map.

Usage:
    python plot_wrf_domains.py namelist.wps
    python plot_wrf_domains.py namelist.wps --output domains.png

"""

import argparse
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt


def _coerce(v):
    """Try int, float, str."""
    v = v.strip().strip("'\"")
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_namelist(path):
    """
    Section-aware Fortran namelist parser.
    Returns a flat dict merging all sections; later sections win on key conflict.
    Keys from &geogrid take priority for domain geometry fields.
    """
    with open(path) as f:
        lines = f.readlines()

    # 1. Strip inline comments and blank lines; join continuation lines
    clean = []
    for line in lines:
        line = re.sub(r'!.*', '', line).strip()
        if line:
            clean.append(line)

    # 2. Re-join into one big string, then split on commas / newlines into tokens
    text = ' '.join(clean)

    # 3. Remove section markers (&name and standalone /)
    text = re.sub(r'&\w+', '', text)
    # Remove standalone slash section terminators (not inside quotes or paths)
    # Replace lines that are just '/' 
    text = re.sub(r'(?<![\'"/\w])/(?![\'"/\w])', ' ', text)
    # Also handle trailing / after last value in a section
    text = re.sub(r'\s*/\s*(?=[a-zA-Z_]|\Z)', ' ', text)

    # 4. Parse key = value pairs, where value may contain commas (multi-value lists)
    #    Strategy: split on `word =` boundaries
    data = {}
    # Find all assignments: key = val [, val ...] up to next key= or end
    # Use a lookahead for the next `word =` pattern
    pattern = re.compile(
        r'([a-zA-Z_]\w*)\s*=\s*(.*?)(?=\s+[a-zA-Z_]\w*\s*=|$)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        key = m.group(1).strip().lower()
        raw_val = m.group(2).strip().rstrip(',').strip()
        if not raw_val:
            continue
        parts = [_coerce(v) for v in raw_val.split(',')]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        data[key] = parts[0] if len(parts) == 1 else parts

    return data


def get_list(data, key, n):
    """Return a list of length n for a namelist key (scalar → repeated)."""
    val = data.get(key, [0] * n)
    if not isinstance(val, list):
        val = [val]
    # Pad / trim to n
    while len(val) < n:
        val.append(val[-1])
    return val[:n]


# ── Domain geometry ────────────────────────────────────────────────────────────

def wrf_proj(nml):
    """Return a Cartopy CRS matching the WPS map_proj."""
    proj = str(nml.get('map_proj', 'lambert')).lower()
    ref_lat = float(nml.get('ref_lat', 0))
    ref_lon = float(nml.get('ref_lon', 0))
    truelat1 = float(nml.get('truelat1', ref_lat))
    truelat2 = float(nml.get('truelat2', truelat1))
    stand_lon = float(nml.get('stand_lon', ref_lon))

    if proj in ('lambert',):
        return ccrs.LambertConformal(
            central_longitude=stand_lon,
            central_latitude=truelat1,
            standard_parallels=(truelat1, truelat2),
        )
    elif proj in ('mercator',):
        return ccrs.Mercator(central_longitude=stand_lon)
    elif proj in ('polar', 'polar-stereographic'):
        return ccrs.Stereographic(
            central_latitude=90 if truelat1 >= 0 else -90,
            central_longitude=stand_lon,
        )
    elif proj in ('lat-lon', 'regular_ll'):
        return ccrs.PlateCarree()
    else:
        print(f"[warn] Unknown map_proj '{proj}', defaulting to LambertConformal.")
        return ccrs.LambertConformal(central_longitude=stand_lon,
                                      central_latitude=truelat1)


def domain_corners(nml, d, proj, geo_crs):
    """
    Return the four (lon, lat) corners of domain d (1-based) in geographic coords.
    Strategy: place domain centre in projection space, compute half-widths, back-project.
    """
    max_dom = int(nml.get('max_dom', 1))

    e_we   = get_list(nml, 'e_we',   max_dom)
    e_sn   = get_list(nml, 'e_sn',   max_dom)
    dx_d1  = float(nml.get('dx', 30000))
    dy_d1  = float(nml.get('dy', dx_d1))

    # Parent grid ratio chain → effective dx/dy for domain d
    parent_grid_ratio = get_list(nml, 'parent_grid_ratio', max_dom)
    i_parent_start    = get_list(nml, 'i_parent_start',    max_dom)
    j_parent_start    = get_list(nml, 'j_parent_start',    max_dom)

    # Compute dx for each domain
    dx = [dx_d1]
    dy = [dy_d1]
    for k in range(1, max_dom):
        dx.append(dx[k-1] / parent_grid_ratio[k])
        dy.append(dy[k-1] / parent_grid_ratio[k])

    # Domain 1 centre is ref_lat / ref_lon
    ref_lat = float(nml.get('ref_lat', 0))
    ref_lon = float(nml.get('ref_lon', 0))

    # Project ref point
    pt = proj.transform_point(ref_lon, ref_lat, geo_crs)

    # Domain 1 total extent in metres
    nx1 = e_we[0] - 1
    ny1 = e_sn[0] - 1

    # Centre of domain 1 in projection space
    cx = [pt[0]]
    cy = [pt[1]]

    # Propagate centre for nested domains
    for k in range(1, max_dom):
        # i_parent_start / j_parent_start are 1-based indices on the parent grid
        parent = 0  # simplification: all nest in d1 for offset calc
        # Offset of nest centre from parent SW corner
        nx_nest = e_we[k] - 1
        ny_nest = e_sn[k] - 1
        # SW corner of domain 1 in proj space
        sw_x = cx[0] - nx1 / 2 * dx[0]
        sw_y = cy[0] - ny1 / 2 * dy[0]
        # Start of this nest on parent
        istart = i_parent_start[k] - 1  # 0-based
        jstart = j_parent_start[k] - 1
        # SW corner of nest in proj
        nest_sw_x = sw_x + istart * dx[0]
        nest_sw_y = sw_y + jstart * dy[0]
        cx.append(nest_sw_x + nx_nest / 2 * dx[k])
        cy.append(nest_sw_y + ny_nest / 2 * dy[k])

    i = d - 1
    half_x = (e_we[i] - 1) / 2 * dx[i]
    half_y = (e_sn[i] - 1) / 2 * dy[i]

    corners_proj = [
        (cx[i] - half_x, cy[i] - half_y),  # SW
        (cx[i] + half_x, cy[i] - half_y),  # SE
        (cx[i] + half_x, cy[i] + half_y),  # NE
        (cx[i] - half_x, cy[i] + half_y),  # NW
    ]

    corners_geo = [geo_crs.transform_point(x, y, proj) for x, y in corners_proj]
    return corners_geo  # list of (lon, lat)


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6',
          '#1ABC9C', '#E67E22', '#34495E']


# Tile zoom level: higher = more detail but slower to fetch.
# Tune this based on domain size. Auto-selected below if not overridden.
DEFAULT_TILE_ZOOM = None  # None = auto

TILE_SOURCES = [
    # (name, tiler_factory)  tried in order until one succeeds
    ('OSM',             lambda: cimgt.OSM()),
    ('GoogleTerrain',   lambda: cimgt.GoogleTiles(style='terrain')),
    ('GoogleStreet',    lambda: cimgt.GoogleTiles(style='street')),
]


def pick_zoom(extent_deg_wide):
    """Auto-select tile zoom level from domain width in degrees."""
    if extent_deg_wide > 60:   return 4
    if extent_deg_wide > 30:   return 5
    if extent_deg_wide > 15:   return 6
    if extent_deg_wide > 7:    return 7
    return 8


def add_tile_background(ax, geo_crs, extent, zoom=None):
    """
    Try each tile source in order. Fall back to stock_img if all fail.
    Returns the name of the source used.
    """
    import urllib.request, warnings

    if zoom is None:
        width_deg = extent[1] - extent[0]
        zoom = pick_zoom(width_deg)

    for name, factory in TILE_SOURCES:
        try:
            tiler = factory()
            # Quick connectivity check before adding to axes
            test_url = tiler._image_url((0, 0, zoom))
            req = urllib.request.Request(test_url,
                                         headers={'User-Agent': 'Mozilla/5.0'})
            urllib.request.urlopen(req, timeout=5)
            # Connection works — add to axes
            ax.add_image(tiler, zoom)
            print(f"[map] Using {name} tiles at zoom={zoom}")
            return name
        except Exception:
            continue

    # Nothing worked — use bundled stock image
    print("[map] Tile servers unreachable, falling back to stock_img.")
    ax.stock_img()
    return 'stock_img'


def plot_domains(nml, output=None, dpi=120, zoom=None):
    max_dom = int(nml.get('max_dom', 1))
    proj    = wrf_proj(nml)
    geo_crs = ccrs.PlateCarree()

    # Collect all corners to determine map extent
    all_corners = []
    domain_corners_list = []
    for d in range(1, max_dom + 1):
        c = domain_corners(nml, d, proj, geo_crs)
        domain_corners_list.append(c)
        all_corners.extend(c)

    lons = [c[0] for c in all_corners]
    lats = [c[1] for c in all_corners]
    lon_pad = (max(lons) - min(lons)) * 0.15 + 2
    lat_pad = (max(lats) - min(lats)) * 0.15 + 2
    extent = [min(lons) - lon_pad, max(lons) + lon_pad,
              min(lats) - lat_pad, max(lats) + lat_pad]

    # Choose a nice display projection centred on domain 1
    ref_lon = float(nml.get('ref_lon', np.mean(lons)))
    ref_lat = float(nml.get('ref_lat', np.mean(lats)))
    map_proj = ccrs.LambertConformal(central_longitude=ref_lon,
                                      central_latitude=ref_lat)

    fig, ax = plt.subplots(figsize=(11, 8),
                           subplot_kw={'projection': map_proj})
    ax.set_extent(extent, crs=geo_crs)

    # Background: try tile imagery (OSM → Google), fall back to stock_img
    bg = add_tile_background(ax, geo_crs, extent, zoom=zoom)
    grid_color = 'white' if bg != 'stock_img' else 'gray'
    ax.gridlines(draw_labels=True, linewidth=0.5, color=grid_color,
                 alpha=0.8, linestyle='--')

    legend_patches = []
    for d, corners in enumerate(domain_corners_list, start=1):
        color = COLORS[(d - 1) % len(COLORS)]
        lw    = 2.5 if d == 1 else 1.8
        ls    = '-'  if d == 1 else '--'
        alpha = 0.25 if d == 1 else 0.15

        # Close the polygon
        poly_lons = [c[0] for c in corners] + [corners[0][0]]
        poly_lats = [c[1] for c in corners] + [corners[0][1]]

        ax.plot(poly_lons, poly_lats, color=color, linewidth=lw,
                linestyle=ls, transform=geo_crs, zorder=5)
        ax.fill(poly_lons, poly_lats, color=color, alpha=alpha,
                transform=geo_crs, zorder=4)

        # Label in the centre of the domain
        clon = np.mean([c[0] for c in corners])
        clat = np.mean([c[1] for c in corners])
        ax.text(clon, clat, f'd{d:02d}', transform=geo_crs,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=color, zorder=6,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none',
                          boxstyle='round,pad=0.2'))

        # Build legend label with domain info
        e_we   = get_list(nml, 'e_we',   max_dom)[d-1]
        e_sn   = get_list(nml, 'e_sn',   max_dom)[d-1]
        dx_d1  = float(nml.get('dx', 30000))
        parent_grid_ratio = get_list(nml, 'parent_grid_ratio', max_dom)
        dx_arr = [dx_d1]
        for k in range(1, max_dom):
            dx_arr.append(dx_arr[k-1] / parent_grid_ratio[k])
        res_km = dx_arr[d-1] / 1000
        label = (f'd{d:02d}  {e_we}×{e_sn} pts  '
                 f'Δx={res_km:.1f} km')
        legend_patches.append(mpatches.Patch(facecolor=color, alpha=0.7, label=label))

    map_proj_str = str(nml.get('map_proj', 'lambert'))
    ax.set_title(f'WRF Domains  |  map_proj: {map_proj_str}  |  '
                 f'max_dom: {max_dom}',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(handles=legend_patches, loc='lower left',
              fontsize=9, framealpha=0.85, title='Domain info')

    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    print(f"Saved → {output}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Plot WRF domains from a namelist.wps file.')
    parser.add_argument('namelist', help='Path to namelist.wps')
    parser.add_argument('--output', '-o', default=None,
                        help='Output image file (e.g. domains.png). '
                             'Defaults to wrf_domains.png in the current directory.')
    parser.add_argument('--dpi', type=int, default=120,
                        help='Image resolution in DPI (default: 120)')
    parser.add_argument('--zoom', type=int, default=None,
                        help='Tile zoom level (default: auto-selected by domain size). '
                             'Higher = more detail, slower. Typical range 4-9.')
    args = parser.parse_args()

    # Default output: wrf_domains.png in the current working directory
    output = args.output or 'wrf_domains.png'

    nml = parse_namelist(args.namelist)
    print(f"Parsed namelist: max_dom={nml.get('max_dom', 1)}, "
          f"map_proj={nml.get('map_proj', '?')}")
    plot_domains(nml, output=output, dpi=args.dpi, zoom=args.zoom)


if __name__ == '__main__':
    main()
