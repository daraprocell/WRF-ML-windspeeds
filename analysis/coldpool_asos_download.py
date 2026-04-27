#!/usr/bin/env python3
"""
ASOS_download.py
----------------
Download full raw ASOS 5-minute time series for WRF cold pool analysis.

Saves two files:
  1. {event}_asos_timeseries.csv  — full raw time series, all stations, one row per observation
  2. {event}_asos_summary.csv     — peak statistics per station (for ML bias correction)

Variables downloaded (all needed for cold pool analysis):
  - tmpf    : temperature (°F)           → cold pool temperature drop
  - dwpf    : dewpoint (°F)              → moisture signature
  - sknt    : sustained wind (knots)     → wind speed
  - gust    : wind gust (knots)          → peak gust
  - drct    : wind direction (degrees)   → direction shift at outflow boundary
  - mslp    : sea level pressure (mb)    → pressure surge
  - wxcodes : weather codes              → precipitation/convection indicator

Example use:
    python ASOS_download.py --event houston --start "2024-05-16 00:00" --end "2024-05-17 06:00"
    python ASOS_download.py --event iowa    --start "2024-07-11 00:00" --end "2024-07-12 06:00"
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

# Unit conversions
KT_TO_MS   = 0.514444   # knots → m/s
F_TO_C     = lambda f: (f - 32) * 5 / 9  # °F → °C

# ---------------------------------------------------------------------------
# Station definitions
# ---------------------------------------------------------------------------

STATION_LISTS = {
    'houston': [
        'KHOU', 'KIAH', 'KGLS', 'KDWH', 'KSGR', 'KTME',
        'KCLL', 'KBPT', 'KLCH', 'KLFT', 'KMSY', 'KNEW', 'KBTR'
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
    'KHOU': {'name': 'Houston Hobby',          'lat': 29.65,  'lon': -95.28,  'elev': 14},
    'KIAH': {'name': 'Bush Intercontinental',  'lat': 29.98,  'lon': -95.34,  'elev': 29},
    'KGLS': {'name': 'Galveston',              'lat': 29.27,  'lon': -94.86,  'elev': 2},
    'KDWH': {'name': 'David Wayne Hooks',      'lat': 30.06,  'lon': -95.55,  'elev': 46},
    'KSGR': {'name': 'Sugar Land',             'lat': 29.62,  'lon': -95.66,  'elev': 25},
    'KTME': {'name': 'Houston Executive',      'lat': 29.81,  'lon': -95.90,  'elev': 34},
    'KCLL': {'name': 'College Station',        'lat': 30.59,  'lon': -96.36,  'elev': 96},
    'KBPT': {'name': 'Beaumont',               'lat': 30.07,  'lon': -94.02,  'elev': 5},
    'KLCH': {'name': 'Lake Charles',           'lat': 30.13,  'lon': -93.22,  'elev': 3},
    'KLFT': {'name': 'Lafayette',              'lat': 30.21,  'lon': -91.99,  'elev': 13},
    'KMSY': {'name': 'New Orleans Intl',       'lat': 29.99,  'lon': -90.26,  'elev': 1},
    'KNEW': {'name': 'New Orleans Lakefront',  'lat': 30.04,  'lon': -90.03,  'elev': 3},
    'KBTR': {'name': 'Baton Rouge',            'lat': 30.53,  'lon': -91.15,  'elev': 21},
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
    'KBDU': {'name': 'Boulder',                'lat': 40.04,  'lon': -105.23, 'elev': 1634},
    'KBJC': {'name': 'Broomfield/Jeffco',      'lat': 39.91,  'lon': -105.12, 'elev': 1724},
    'KFNL': {'name': 'Fort Collins',           'lat': 40.45,  'lon': -105.01, 'elev': 1529},
    'KDEN': {'name': 'Denver Intl',            'lat': 39.86,  'lon': -104.67, 'elev': 1655},
    'KCOS': {'name': 'Colorado Springs',       'lat': 38.81,  'lon': -104.70, 'elev': 1881},
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_station(station, start_time, end_time, retries=3):
    """
    Download full 5-minute ASOS time series from Iowa Environmental Mesonet.
    Returns a DataFrame with one row per observation, or None on failure.
    """
    url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

    # Request all variables needed for cold pool analysis
    params = {
        'station':  station,
        'data':     'tmpf,dwpf,sknt,gust,drct,mslp,wxcodes,feel',
        'year1':    start_time.year,
        'month1':   start_time.month,
        'day1':     start_time.day,
        'year2':    end_time.year,
        'month2':   end_time.month,
        'day2':     end_time.day,
        'tz':       'UTC',
        'format':   'onlycomma',
        'latlon':   'yes',
        'elev':     'yes',
        'missing':  'M',
        'trace':    'T',
        'direct':   'no',
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), na_values=['M', '', 'T'])
            if df.empty:
                print(f"  {station}: no data returned")
                return None
            df['valid'] = pd.to_datetime(df['valid'])
            df = df[(df['valid'] >= start_time) & (df['valid'] <= end_time)]
            if df.empty:
                print(f"  {station}: no data in time range")
                return None
            print(f"  {station}: {len(df)} observations downloaded")
            return df
        except Exception as e:
            print(f"  {station}: attempt {attempt+1} failed — {e}")
            time.sleep(2)
    return None


def process_timeseries(df, station, start_time, end_time):
    """
    Clean and enrich raw ASOS data with derived cold pool variables.
    Returns a tidy DataFrame with all variables in SI units.
    """
    if df is None or df.empty:
        return None

    meta = STATION_METADATA.get(station, {})

    # Use metadata lat/lon if IEM didn't return them
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

    # Temperature — convert °F to °C
    out['temp_c']  = df['tmpf'].apply(lambda x: F_TO_C(x) if pd.notna(x) else np.nan)
    out['dwpt_c']  = df['dwpf'].apply(lambda x: F_TO_C(x) if pd.notna(x) else np.nan)

    # Winds — convert knots to m/s
    out['wspd_ms'] = df['sknt'] * KT_TO_MS if 'sknt' in df.columns else np.nan
    out['gust_ms'] = df['gust'] * KT_TO_MS if 'gust' in df.columns else np.nan
    out['wdir']    = df['drct'] if 'drct' in df.columns else np.nan

    # Pressure
    out['mslp_mb'] = df['mslp'] if 'mslp' in df.columns else np.nan

    # Weather codes (for identifying convective passage)
    out['wxcodes']  = df['wxcodes'] if 'wxcodes' in df.columns else ''

    # --- Derived cold pool variables ---

    # Temperature anomaly: deviation from pre-event mean (first 3 hours)
    pre_event_end = start_time + pd.Timedelta(hours=3)
    pre_mask = out['valid'] <= pre_event_end
    pre_mean_temp = out.loc[pre_mask, 'temp_c'].mean()
    out['temp_anomaly_c'] = out['temp_c'] - pre_mean_temp

    # Pressure anomaly: deviation from pre-event mean
    pre_mean_pres = out.loc[pre_mask, 'mslp_mb'].mean()
    out['pres_anomaly_mb'] = out['mslp_mb'] - pre_mean_pres

    # Rolling 1-hour max gust (12 obs at 5-min resolution)
    out['max_1hr_gust'] = out['gust_ms'].rolling(12, min_periods=3).max()

    # Cold pool passage flag: temp drop > 3°C in 30 min (6 obs)
    temp_change = out['temp_c'].diff(6)
    out['cold_pool_passage'] = (temp_change <= -3.0).astype(int)

    # Time of cold pool passage
    passage_times = out.loc[out['cold_pool_passage'] == 1, 'valid']
    out['cold_pool_arrival'] = passage_times.min() if len(passage_times) > 0 else pd.NaT

    return out


def compute_summary(timeseries_df, station, start_time, end_time):
    """
    Compute per-station summary statistics from the full time series.
    Compatible with the ML bias correction pipeline.
    """
    df = timeseries_df[timeseries_df['station'] == station].copy()
    if df.empty:
        return None

    gusts = df['gust_ms'].dropna()
    winds = df['wspd_ms'].dropna()

    # Cold pool stats
    cp_rows = df[df['cold_pool_passage'] == 1]
    if len(cp_rows) > 0:
        cp_arrival     = df['cold_pool_arrival'].iloc[0]
        cp_temp_drop   = df.loc[cp_rows.index, 'temp_anomaly_c'].min()
        cp_pres_surge  = df.loc[cp_rows.index, 'pres_anomaly_mb'].max()
    else:
        cp_arrival     = pd.NaT
        cp_temp_drop   = np.nan
        cp_pres_surge  = np.nan

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
        'pre_event_mean_temp':  df.loc[df['valid'] <= start_time + pd.Timedelta(hours=3),
                                       'temp_c'].mean(),
        'pct_missing_wind':     (df['wspd_ms'].isna().sum() / len(df)) * 100,
        'pct_missing_gust':     (df['gust_ms'].isna().sum() / len(df)) * 100,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download full ASOS time series for WRF cold pool analysis')
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
    print(f"Downloading {args.event} ASOS data for {len(stations)} stations")
    print(f"Period: {start_time} → {end_time} UTC\n")

    all_timeseries = []
    all_summaries  = []

    for station in stations:
        raw = download_station(station, start_time, end_time)
        ts  = process_timeseries(raw, station, start_time, end_time)

        if ts is not None:
            all_timeseries.append(ts)
            summary = compute_summary(ts, station, start_time, end_time)
            if summary:
                all_summaries.append(summary)

        time.sleep(0.5)  # be polite to IEM servers

    if not all_timeseries:
        print("ERROR: No data downloaded successfully")
        return 1

    # Save full time series — one CSV, all stations
    ts_df = pd.concat(all_timeseries, ignore_index=True)
    ts_df = ts_df.sort_values(['station', 'valid']).reset_index(drop=True)
    ts_file = output_dir / f'{args.event}_asos_timeseries.csv'
    ts_df.to_csv(ts_file, index=False)
    print(f"\nTime series saved: {ts_file}  ({len(ts_df)} rows, {ts_df['station'].nunique()} stations)")

    # Save summary — compatible with ML pipeline
    sum_df = pd.DataFrame(all_summaries)
    sum_file = output_dir / f'{args.event}_asos_summary.csv'
    sum_df.to_csv(sum_file, index=False)
    print(f"Summary saved:     {sum_file}  ({len(sum_df)} stations)")

    # Print quick overview
    print(f"\n--- {args.event.upper()} Cold Pool Summary ---")
    for _, row in sum_df.iterrows():
        cp = f"CP arrival: {row['cold_pool_arrival']}" if pd.notna(row['cold_pool_arrival']) else "CP: not detected"
        print(f"  {row['station']:6s} | peak gust: {row['peak_gust']:.1f} m/s | "
              f"temp drop: {row['cold_pool_temp_drop']:.1f}°C | {cp}")

    return 0


if __name__ == '__main__':
    exit(main())
