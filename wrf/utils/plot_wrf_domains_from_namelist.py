#!/usr/bin/env python3
"""Plot WRF domains from a namelist.wps file."""

import argparse
import f90nml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

COLORS = ['red', 'green', 'blue', 'orange', 'purple']


def get_list(section, key, n):
    """Return list of length n; scalars are repeated, short lists are padded."""
    val = section[key]
    if not isinstance(val, list):
        val = [val]
    while len(val) < n:
        val.append(val[-1])
    return val[:n]


def domain_corners(geo, share, proj, geo_crs):
    """Yield (lon, lat) corners for each domain as a list of 4 points."""
    max_dom = share['max_dom']
    e_we = get_list(geo, 'e_we', max_dom)
    e_sn = get_list(geo, 'e_sn', max_dom)
    pgr = get_list(geo, 'parent_grid_ratio', max_dom)
    istart = get_list(geo, 'i_parent_start', max_dom)
    jstart = get_list(geo, 'j_parent_start', max_dom)

    dx = [geo['dx']]
    dy = [geo['dy']]
    for k in range(1, max_dom):
        dx.append(dx[k-1] / pgr[k])
        dy.append(dy[k-1] / pgr[k])

    # SW corner of d01 in projection coordinates
    cx0, cy0 = proj.transform_point(geo['ref_lon'], geo['ref_lat'], geo_crs)
    sw_x = [cx0 - (e_we[0] - 1) / 2 * dx[0]]
    sw_y = [cy0 - (e_sn[0] - 1) / 2 * dy[0]]

    # Each nest's SW corner sits at (istart-1, jstart-1) on its parent grid
    for k in range(1, max_dom):
        sw_x.append(sw_x[0] + (istart[k] - 1) * dx[0])
        sw_y.append(sw_y[0] + (jstart[k] - 1) * dy[0])

    domains = []
    for k in range(max_dom):
        w = (e_we[k] - 1) * dx[k]
        h = (e_sn[k] - 1) * dy[k]
        corners_proj = [
            (sw_x[k],     sw_y[k]),
            (sw_x[k] + w, sw_y[k]),
            (sw_x[k] + w, sw_y[k] + h),
            (sw_x[k],     sw_y[k] + h),
        ]
        domains.append([geo_crs.transform_point(x, y, proj) for x, y in corners_proj])
    return domains, dx, e_we, e_sn


def plot_domains(nml_path, output, dpi=120):
    nml = f90nml.read(nml_path)
    geo = nml['geogrid']
    share = nml['share']

    proj = ccrs.LambertConformal(
        central_longitude=geo['stand_lon'],
        central_latitude=geo['truelat1'],
        standard_parallels=(geo['truelat1'], geo.get('truelat2', geo['truelat1'])),
    )
    geo_crs = ccrs.PlateCarree()

    domains, dx, e_we, e_sn = domain_corners(geo, share, proj, geo_crs)

    lons = [c[0] for d in domains for c in d]
    lats = [c[1] for d in domains for c in d]
    pad_x = (max(lons) - min(lons)) * 0.15 + 2
    pad_y = (max(lats) - min(lats)) * 0.15 + 2
    extent = [min(lons) - pad_x, max(lons) + pad_x,
              min(lats) - pad_y, max(lats) + pad_y]

    fig, ax = plt.subplots(figsize=(11, 8), subplot_kw={'projection': proj})
    ax.set_extent(extent, crs=geo_crs)
    ax.add_feature(cfeature.LAND, facecolor='wheat')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')     # Great Lakes
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor='steelblue')
    ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                 alpha=0.6, linestyle='--')

    patches = []
    for d, corners in enumerate(domains, start=1):
        color = COLORS[(d - 1) % len(COLORS)]
        lw = 2.5 if d == 1 else 1.8
        ls = '-' if d == 1 else '--'

        poly_lons = [c[0] for c in corners] + [corners[0][0]]
        poly_lats = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(poly_lons, poly_lats, color=color, lw=lw, ls=ls,
                transform=geo_crs, zorder=5)
        ax.fill(poly_lons, poly_lats, color=color, alpha=0.18,
                transform=geo_crs, zorder=4)

        clon = sum(c[0] for c in corners) / 4
        clat = sum(c[1] for c in corners) / 4
        ax.text(clon, clat, f'd{d:02d}', transform=geo_crs,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=color, zorder=6,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                          boxstyle='round,pad=0.2'))

        patches.append(mpatches.Patch(
            facecolor=color, alpha=0.7,
            label=f'd{d:02d}  {e_we[d-1]}×{e_sn[d-1]} pts  Δx={dx[d-1]/1000:.1f} km'
        ))

    ax.set_title(f'WRF Domains  |  max_dom: {share["max_dom"]}',
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(handles=patches, loc='lower left', fontsize=9,
              framealpha=0.85, title='Domain info')

    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.show()


def main():
    p = argparse.ArgumentParser(description='Plot WRF domains from namelist.wps.')
    p.add_argument('namelist', help='Path to namelist.wps')
    p.add_argument('-o', '--output', default='wrf_domains.png')
    p.add_argument('--dpi', type=int, default=120)
    args = p.parse_args()
    plot_domains(args.namelist, args.output, args.dpi)


if __name__ == '__main__':
    main()