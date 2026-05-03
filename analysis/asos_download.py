#!/usr/bin/env python3
"""
asos_download.py
----------------
Download ASOS data for WRF wind and cold pool comparison.

Saves two files (both useful, different downstream consumers):
  1. {event}_asos_timeseries.csv  — full 5-min time series, one row per observation
                                     (used by coldpool_analysis.py, time series plots)
  2. {event}_asos_summary.csv     — peak statistics per station, one row per station
                                     (used by wind_swath_comparison.py, ML pipeline)

Variables downloaded (all needed for cold pool + wind analysis):
  - tmpf    : temperature (°F)           → cold pool temperature drop
  - dwpf    : dewpoint (°F)              → moisture signature
  - sknt    : sustained wind (knots)     → wind speed
  - gust    : wind gust (knots)          → peak gust (used for swath comparison)
  - drct    : wind direction (degrees)   → direction shift at outflow boundary
  - mslp    : sea level pressure (mb)    → pressure surge
  - wxcodes : weather codes              → precipitation/convection indicator

Example use:
    python asos_download.py --event houston --start "2024-05-16 00:00" --end "2024-05-17 06:00"
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from io import StringIO
from pathlib import Path
import json
import time

KT_TO_MS = 0.514444   # knots → m/s
F_TO_C   = lambda f: (f - 32) * 5 / 9   # °F → °C

# Houston event stations:
#   - 13 original Houston-area stations (used for initial bias analysis)
#   - 6 extended stations near WRF-simulated max wind location (~31.1°N, -94.1°W)
#     KJAS is the closest ASOS to the WRF max 
STATION_LISTS = {
    'houston': [
        # Original 13 Houston-area stations
        'KHOU', 'KIAH', 'KGLS', 'KDWH', 'KSGR', 'KTME',
        'KCLL', 'KBPT', 'KLCH', 'KLFT', 'KMSY', 'KNEW', 'KBTR',
        # Extended stations near WRF-simulated max wind location
        'KJAS',   # Jasper, TX -- closest to WRF max at 31.1°N, -94.1°W
        'KLFK',   # Lufkin, TX -- in WRF storm path
        'KCXO',   # Conroe, TX -- between Houston and WRF max
        'KAEX',   # Alexandria, LA -- east of WRF max
        'K6R3',   # Cleveland, TX -- north of Houston
        'KPSX',   # Palacios, TX -- southwest of Houston
    ],
    'iowa': [
        'KDSM', 'KCID', 'KDVN', 'KMCW', 'KOTM',
        'KORD', 'KMDW', 'KMLI', 'KIND', 'KCVG'
    ],
    'midwest': [
        'KORD', 'KIND', 'KCVG', 'KCMH', 'KPIT',
        'KDAY', 'KFWA', 'KSBN', 'KGRR', 'KLAN'
    ],
    'colorado': [
        'KBDU', 'KBJC', 'KFNL', 'KDEN', 'KCOS'
    ],
}

STATION_METADATA = {
    # Houston original 13
    'KHOU': {'name': 'Houston Hobby',          'lat': 29.6375, 'lon': -95.2824, 'elev': 14},
    'KIAH': {'name': 'Bush Intercontinental',  'lat': 29.9844, 'lon': -95.3607, 'elev': 29},
    'KGLS': {'name': 'Galveston',              'lat': 29.2654, 'lon': -94.8604, 'elev': 2},
    'KDWH': {'name': 'David Wayne Hooks',      'lat': 30.0680, 'lon': -95.5562, 'elev': 46},
    'KSGR': {'name': 'Sugar Land',             'lat': 29.6220, 'lon': -95.6570, 'elev': 25},
    'KTME': {'name': 'Houston Executive',      'lat': 29.8050, 'lon': -95.8975, 'elev': 34},
    'KCLL': {'name': 'College Station',        'lat': 30.5886, 'lon': -96.3638, 'elev': 96},
    'KBPT': {'name': 'Beaumont',               'lat': 29.9508, 'lon': -94.0207, 'elev': 5},
    'KLCH': {'name': 'Lake Charles',           'lat': 30.1261, 'lon': -93.2284, 'elev': 3},
    'KLFT': {'name': 'Lafayette',              'lat': 30.2053, 'lon': -91.9876, 'elev': 13},
    'KMSY': {'name': 'New Orleans Intl',       'lat': 29.9934, 'lon': -90.2580, 'elev': 1},
    'KNEW': {'name': 'New Orleans Lakefront',  'lat': 30.0420, 'lon': -90.0283, 'elev': 3},
    'KBTR': {'name': 'Baton Rouge',            'lat': 30.5333, 'lon': -91.1496, 'elev': 21},

    # Houston extended stations
    'KJAS': {'name': 'Jasper Bell Field',      'lat': 30.8857, 'lon': -94.0349, 'elev': 65},
    'KLFK': {'name': 'Lufkin Angelina Co.',    'lat': 31.2340, 'lon': -94.7500, 'elev': 88},
    'KCXO': {'name': 'Conroe Lone Star Exec.', 'lat': 30.3520, 'lon': -95.4150, 'elev': 75},
    'KAEX': {'name': 'Alexandria Intl',        'lat': 31.3274, 'lon': -92.5499, 'elev': 27},
    'K6R3': {'name': 'Cleveland Municipal',    'lat': 30.3470, 'lon': -94.7080, 'elev': 47},
    'KPSX': {'name': 'Palacios Municipal',     'lat': 28.7275, 'lon': -96.2510, 'elev': 4},

    # Iowa
    'KDSM': {'name': 'Des Moines',             'lat': 41.53,  'lon': -93.66,  'elev': 294},
    'KCID': {'name': 'Cedar Rapids',           'lat': 41.88,  'lon': -91.71,  'elev': 265},
    'KDVN': {'name': 'Davenport',              'lat': 41.61,  'lon': -90.59,  'elev': 229},
    'KMCW': {'name': 'Mason City',             'lat': 43.16,  'lon': -93.33,  'elev': 373},
    'KOTM': {'name': 'Ottumwa',                'lat': 41.11,  'lon': -92.45,  'elev': 256},
    'KORD': {'name': "Chicago O'Hare",         'lat': 41.98,  'lon': -87.90,  'elev': 205},
    'KMDW': {'name': 'Chicago Midway',         'lat': 41.79,  'lon': -87.75,  'elev': 188},
    'KMLI': {'name': 'Moline',                 'lat': 41.45,  'lon': -90.51,  'elev': 181},
    'KIND': {'name': 'Indianapolis',           'lat': 39.73,  'lon': -86.27,  'elev': 241},
    'KCVG': {'name': 'Cincinnati',             'lat': 39.05,  'lon': -84.67,  'elev': 270},
    'KCMH': {'name': 'Columbus',               'lat': 39.99,  'lon': -82.89,  'elev': 247},
    'KPIT': {'name': 'Pittsburgh',             'lat': 40.49,  'lon': -80.23,  'elev': 373},
    'KDAY': {'name': 'Dayton',                 'lat': 39.90,  'lon': -84.22,  'elev': 306},
    'KFWA': {'name': 'Fort Wayne',             'lat': 40.98,  'lon': -85.20,  'elev': 252},
    'KSBN': {'name': 'South Bend',             'lat': 41.71,  'lon': -86.32,  'elev': 236},
    'KGRR': {'name': 'Grand Rapids',           'lat': 42.88,  'lon': -85.52,  'elev': 245},
    'KLAN': {'name': 'Lansing',                'lat': 42.78,  'lon': -84.59,  'elev': 283},

    # Colorado
    'KBDU': {'name': 'Boulder',                'lat': 40.04,  'lon': -105.23, 'elev': 1634},
    'KBJC': {'name': 'Broomfield/Jeffco',      'lat': 39.91,  'lon': -105.12, 'elev': 1724},
    'KFNL': {'name': 'Fort Collins',           'lat': 40.45,  'lon': -105.01, 'elev': 1529},
    'KDEN': {'name': 'Denver Intl',            'lat': 39.86,  'lon': -104.67, 'elev': 1655},
    'KCOS': {'name': 'Colorado Springs',       'lat': 38.81,  'lon': -104.70, 'elev': 1881},
}

def download_station(station, start_time, end_time, retries=3):
    """Download full 5-min ASOS time series from Iowa Environmental Mesonet."""
    url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

    params = {
        'station':  station,
        'data':     'tmpf,dwpf,sknt,gust,drct,mslp,wxcodes,feel',
        'year1':    start_time.year, 'month1': start_time.month, 'day1': start_time.day,
        'year2':    end_time.year,   'month2': end_time.month,   'day2': end_time.day,
        'tz':       'UTC',
        'format':   'onlycomma',
        'latlon':   'yes',
        'elev':     'yes',
        'missing':  'M',
        'trace':    'T',
        'direct':   'no',
    }

    # Error handling
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), na_values=['M', '', 'T'])
            if df.empty:
                print(f"{station}: no data returned")
                return None
            df['valid'] = pd.to_datetime(df['valid'])
            df = df[(df['valid'] >= start_time) & (df['valid'] <= end_time)]
            if df.empty:
                print(f"{station}: no data in time range")
                return None
            print(f"  {station}: {len(df)} observations downloaded")
            return df
        except Exception as e:
            print(f"{station}: attempt {attempt+1} failed — {e}")
            time.sleep(2)
    return None


def process_timeseries(df, station, start_time, end_time):
    """Clean and enrich raw ASOS data with derived cold pool variables."""
    if df is None or df.empty:
        return None

    meta = STATION_METADATA.get(station, {})

    if 'lat' not in df.columns or df['lat'].isna().all():
        df['lat'] = meta.get('lat', np.nan)
        df['lon'] = meta.get('lon', np.nan)

    out = pd.DataFrame()
    out['valid']   = df['valid']
    out['station'] = station
    out['name']    = meta.get('name', station)
    out['lat']     = df['lat'].iloc[0] if 'lat' in df.columns else meta.get('lat', np.nan)
    out['lon']     = df['lon'].iloc[0] if 'lon' in df.columns else meta.get('lon', np.nan)
    out['elev']    = df['elevation'].iloc[0] if 'elevation' in df.columns else meta.get('elev', np.nan)

    # Convert F to C
    out['temp_c']  = df['tmpf'].apply(lambda x: F_TO_C(x) if pd.notna(x) else np.nan)
    out['dwpt_c']  = df['dwpf'].apply(lambda x: F_TO_C(x) if pd.notna(x) else np.nan)

    # Convert knots to m/s
    out['wspd_ms'] = df['sknt'] * KT_TO_MS if 'sknt' in df.columns else np.nan
    out['gust_ms'] = df['gust'] * KT_TO_MS if 'gust' in df.columns else np.nan
    out['wdir']    = df['drct'] if 'drct' in df.columns else np.nan

    out['mslp_mb'] = df['mslp'] if 'mslp' in df.columns else np.nan
    out['wxcodes'] = df['wxcodes'] if 'wxcodes' in df.columns else ''

    # Derived cold pool variables: temp anomaly relative to first 3 hours
    pre_event_end = start_time + pd.Timedelta(hours=3)
    pre_mask = out['valid'] <= pre_event_end
    pre_mean_temp = out.loc[pre_mask, 'temp_c'].mean()
    out['temp_anomaly_c'] = out['temp_c'] - pre_mean_temp

    pre_mean_pres = out.loc[pre_mask, 'mslp_mb'].mean()
    out['pres_anomaly_mb'] = out['mslp_mb'] - pre_mean_pres

    out['max_1hr_gust'] = out['gust_ms'].rolling(12, min_periods=3).max()

    temp_change = out['temp_c'].diff(6)
    out['cold_pool_passage'] = (temp_change <= -3.0).astype(int)
    passage_times = out.loc[out['cold_pool_passage'] == 1, 'valid']
    out['cold_pool_arrival'] = passage_times.min() if len(passage_times) > 0 else pd.NaT

    return out


def compute_summary(timeseries_df, station):
    """Compute per-station summary statistics from the full time series."""
    df = timeseries_df[timeseries_df['station'] == station].copy()
    if df.empty:
        return None

    gusts = df['gust_ms'].dropna()
    winds = df['wspd_ms'].dropna()

    cp_rows = df[df['cold_pool_passage'] == 1]
    if len(cp_rows) > 0:
        cp_arrival    = df['cold_pool_arrival'].iloc[0]
        cp_temp_drop  = df.loc[cp_rows.index, 'temp_anomaly_c'].min()
        cp_pres_surge = df.loc[cp_rows.index, 'pres_anomaly_mb'].max()
    else:
        cp_arrival    = pd.NaT
        cp_temp_drop  = np.nan
        cp_pres_surge = np.nan

    return {
        'station':              station,
        'name':                 df['name'].iloc[0],
        'lat':                  df['lat'].iloc[0],
        'lon':                  df['lon'].iloc[0],
        'elevation':            df['elev'].iloc[0],
        'n_obs':                len(df),
        'peak_gust':            gusts.max() if len(gusts) > 0 else np.nan,
        'peak_sustained_wind':  winds.max() if len(winds) > 0 else np.nan,
        'time_peak_gust':       df.loc[gusts.idxmax(), 'valid'] if len(gusts) > 0 else pd.NaT,
        'time_peak_sustained':  df.loc[winds.idxmax(), 'valid'] if len(winds) > 0 else pd.NaT,
        'mean_wind':            winds.mean() if len(winds) > 0 else np.nan,
        'cold_pool_arrival':    cp_arrival,
        'cold_pool_temp_drop':  cp_temp_drop,
        'cold_pool_pres_surge': cp_pres_surge,
        'pct_missing_wind':     (df['wspd_ms'].isna().sum() / len(df)) * 100,
        'pct_missing_gust':     (df['gust_ms'].isna().sum() / len(df)) * 100,
    }

def main():
    parser = argparse.ArgumentParser(
        description='Download ASOS data for WRF wind + cold pool analysis')
    parser.add_argument('--event',      required=True,
                        choices=list(STATION_LISTS.keys()),
                        help='Event name')
    parser.add_argument('--start',      required=True,
                        help='Start time UTC: "YYYY-MM-DD HH:MM"')
    parser.add_argument('--end',        required=True,
                        help='End time UTC: "YYYY-MM-DD HH:MM"')
    parser.add_argument('--output-dir', default='data/asos',
                        help='Output directory')
    args = parser.parse_args()

    start_time = pd.to_datetime(args.start)
    end_time   = pd.to_datetime(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = STATION_LISTS[args.event]
    all_timeseries = []
    all_summaries  = []

    for station in stations:
        raw = download_station(station, start_time, end_time)
        ts  = process_timeseries(raw, station, start_time, end_time)

        if ts is not None:
            all_timeseries.append(ts)
            summary = compute_summary(ts, station)
            if summary:
                all_summaries.append(summary)

        time.sleep(0.5)   # be polite to IEM servers

    if not all_timeseries:
        print("ERROR: No data downloaded successfully")
        return 1

    # Save full time series
    ts_df = pd.concat(all_timeseries, ignore_index=True)
    ts_df = ts_df.sort_values(['station', 'valid']).reset_index(drop=True)
    ts_file = output_dir / f'{args.event}_asos_timeseries.csv'
    ts_df.to_csv(ts_file, index=False)

    sum_df = pd.DataFrame(all_summaries)
    sum_df = sum_df.sort_values('peak_gust', ascending=False)
    sum_file = output_dir / f'{args.event}_asos_summary.csv'
    sum_df.to_csv(sum_file, index=False)

    meta_file = output_dir / f'{args.event}_metadata.json'
    metadata = {
        'event':       args.event,
        'start_time':  start_time.isoformat(),
        'end_time':    end_time.isoformat(),
        'n_stations':  len(sum_df),
        'stations':    sum_df['station'].tolist(),
        'downloaded':  datetime.now().isoformat(),
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    for _, row in sum_df.iterrows():
        cp = (f"CP: {row['cold_pool_arrival'].strftime('%H:%MZ')}"
              if pd.notna(row['cold_pool_arrival']) else "CP: not detected")
        peak_gust = row['peak_gust'] if pd.notna(row['peak_gust']) else 0.0
        temp_drop = row['cold_pool_temp_drop'] if pd.notna(row['cold_pool_temp_drop']) else 0.0
        print(f"  {row['station']:6s} | gust: {peak_gust:5.1f} m/s | "
              f"\u0394T: {temp_drop:+5.1f}\u00b0C | {cp}")

    return 0


if __name__ == '__main__':
    exit(main())