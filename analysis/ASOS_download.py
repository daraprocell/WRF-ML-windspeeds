#!/usr/bin/env python3
"""
download_asos_data.py
---------------------
Download and process ASOS station data for WRF wind comparison.

This script:
1. Downloads ASOS 5 minute data for specified time period
2. Computes peak wind gusts and sustained winds
3. Saves processed data

Example use:
    python ASOS_download.py --event houston --start "2024-05-16 06:00" --end "2024-05-17 06:00"
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO
from pathlib import Path
import json


class ASOSDownloader:
    """Download and process ASOS data from Iowa Environmental Mesonet."""
    
    # ASOS stations for each event
    STATION_LISTS = {
        'houston': [
            'KHOU', 'KIAH', 'KGLS', 'KDWH', 'KSGR', 'KTME',  # Houston area
            'KCLL', 'KBPT', 'KLCH', 'KLFT', 'KMSY', 'KNEW', 'KBTR' 
        ],
        'iowa': [
            'KDSM', 'KCID', 'KDVN', 'KMCW', 'KOTM',  # Iowa
            'KORD', 'KMDW', 'KMLI', 'KIND', 'KCVG'   # Illinois/Indiana
        ],
        'june2012': [
            'KORD', 'KIND', 'KCVG', 'KCMH', 'KPIT',  # Midwest
            'KDAY', 'KFWA', 'KSBN', 'KGRR', 'KLAN'
        ],
        'marshall': [
            'KBDU', 'KBJC', 'KFNL', 'KDEN', 'KCOS'  # Colorado Front Range
        ]
    }
    
    # Station metadata 
    STATION_METADATA = {
        'KHOU': {'name': 'Houston Hobby', 'lat': 29.65, 'lon': -95.28, 'elev': 14},
        'KIAH': {'name': 'Bush Intercontinental', 'lat': 29.98, 'lon': -95.34, 'elev': 29},
        'KGLS': {'name': 'Galveston', 'lat': 29.27, 'lon': -94.86, 'elev': 2},
        'KDWH': {'name': 'David Wayne Hooks', 'lat': 30.06, 'lon': -95.55, 'elev': 46},
        'KSGR': {'name': 'Sugar Land', 'lat': 29.62, 'lon': -95.66, 'elev': 25},
        'KTME': {'name': 'Houston Executive', 'lat': 29.81, 'lon': -95.90, 'elev': 34},
        'KCLL': {'name': 'College Station', 'lat': 30.59, 'lon': -96.36, 'elev': 96},
        'KBPT': {'name': 'Beaumont', 'lat': 30.07, 'lon': -94.02, 'elev': 5},
        'KLCH': {'name': 'Lake Charles', 'lat': 30.13, 'lon': -93.22, 'elev': 3},
        'KLFT': {'name': 'Lafayette', 'lat': 30.21, 'lon': -91.99, 'elev': 13},
        'KMSY': {'name': 'New Orleans Intl', 'lat': 29.99, 'lon': -90.26, 'elev': 1},
        'KNEW': {'name': 'New Orleans Lakefront', 'lat': 30.04, 'lon': -90.03, 'elev': 3},
        'KBTR': {'name': 'Baton Rouge', 'lat': 30.53, 'lon': -91.15, 'elev': 21},
        # Iowa stations
        'KDSM': {'name': 'Des Moines', 'lat': 41.53, 'lon': -93.66, 'elev': 294},
        'KCID': {'name': 'Cedar Rapids', 'lat': 41.88, 'lon': -91.71, 'elev': 265},
        'KDVN': {'name': 'Davenport', 'lat': 41.61, 'lon': -90.59, 'elev': 229},
        'KMCW': {'name': 'Mason City', 'lat': 43.16, 'lon': -93.33, 'elev': 373},
        'KOTM': {'name': 'Ottumwa', 'lat': 41.11, 'lon': -92.45, 'elev': 256},
        'KORD': {'name': 'Chicago O\'Hare', 'lat': 41.98, 'lon': -87.90, 'elev': 205},
        'KMDW': {'name': 'Chicago Midway', 'lat': 41.79, 'lon': -87.75, 'elev': 188},
        'KMLI': {'name': 'Moline', 'lat': 41.45, 'lon': -90.51, 'elev': 181},
        'KIND': {'name': 'Indianapolis', 'lat': 41.87, 'lon': -86.27, 'elev': 241},
        'KCVG': {'name': 'Cincinnati', 'lat': 39.05, 'lon': -84.67, 'elev': 270},
        # Colorado stations
        'KBDU': {'name': 'Boulder', 'lat': 40.04, 'lon': -105.23, 'elev': 1634},
        'KBJC': {'name': 'Broomfield/Jeffco', 'lat': 39.91, 'lon': -105.12, 'elev': 1724},
        'KFNL': {'name': 'Fort Collins', 'lat': 40.45, 'lon': -105.01, 'elev': 1529},
        'KDEN': {'name': 'Denver Intl', 'lat': 39.86, 'lon': -104.67, 'elev': 1655},
        'KCOS': {'name': 'Colorado Springs', 'lat': 38.81, 'lon': -104.70, 'elev': 1881},
    }
    
    def __init__(self, event, start_time, end_time, output_dir='data/asos'):
        """
        Initialize ASOS downloader.
        
        Parameters
        ----------
        event : str
            Event name ('houston', 'iowa', 'june2012', 'marshall')
        start_time : str or datetime
            Start time (UTC) in format 'YYYY-MM-DD HH:MM'
        end_time : str or datetime
            End time (UTC)
        output_dir : str
            Directory to save processed data
        """
        self.event = event
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if event not in self.STATION_LISTS:
            raise ValueError(f"Event '{event}' not recognized. Choose from: {list(self.STATION_LISTS.keys())}")
        
        self.stations = self.STATION_LISTS[event]
    
    def download_station_data(self, station):
        """
        Download ASOS data for a single station from Iowa Mesonet.
        
        Uses IEM ASOS download service: https://mesonet.agron.iastate.edu/request/download.phtml
        """
        
        # IEM ASOS API endpoint
        url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
        
        params = {
            'station': station,
            'data': 'all',  
            'year1': self.start_time.year,
            'month1': self.start_time.month,
            'day1': self.start_time.day,
            'year2': self.end_time.year,
            'month2': self.end_time.month,
            'day2': self.end_time.day,
            'tz': 'UTC',
            'format': 'onlycomma',  
            'latlon': 'yes',
            'elev': 'yes',
            'missing': 'M',
            'trace': 'T',
            'direct': 'no',
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), na_values=['M', '', 'T'])
            
            if df.empty:
                print("No data returned")
                return None
            
            df['valid'] = pd.to_datetime(df['valid'])
            df = df[(df['valid'] >= self.start_time) & (df['valid'] <= self.end_time)]
            
            if df.empty:
                print("No data in time range")
                return None
            
            return df
            
        except Exception as e:
            print(" Error downlaoding!")
            return None
    
    def process_station_data(self, df, station):
        """
        Process raw ASOS data to extract relevant wind information.
        
        Returns
        -------
        dict
            Processed station data with peak winds and metadata
        """
        if df is None or df.empty:
            return None
        
        # Extract wind data (knots in ASOS)
        # Convert to m/s: 1 knot = 0.514444 m/s
        KT_TO_MS = 0.514444
        
        # Sustained wind (sknt = wind speed in knots)
        if 'sknt' in df.columns:
            winds = df['sknt'].dropna() * KT_TO_MS
        else:
            winds = pd.Series([], dtype=float)
        
        # Peak gust (gust = wind gust in knots)
        if 'gust' in df.columns:
            gusts = df['gust'].dropna() * KT_TO_MS
        else:
            gusts = pd.Series([], dtype=float)
        
        # Get station metadata
        if 'lat' in df.columns and 'lon' in df.columns:
            lat = df['lat'].iloc[0]
            lon = df['lon'].iloc[0]
            elev = df['elevation'].iloc[0] if 'elevation' in df.columns else np.nan
        else:
            # Fallback to metadata dict
            meta = self.STATION_METADATA.get(station, {})
            lat = meta.get('lat', np.nan)
            lon = meta.get('lon', np.nan)
            elev = meta.get('elev', np.nan)
        
        # Statistics
        result = {
            'station': station,
            'lat': lat,
            'lon': lon,
            'elevation': elev,
            'n_obs': len(df),
            
            'peak_sustained_wind': winds.max() if len(winds) > 0 else np.nan,
            'peak_gust': gusts.max() if len(gusts) > 0 else np.nan,
            
            'time_peak_sustained': df.loc[winds.idxmax(), 'valid'] if len(winds) > 0 else pd.NaT,
            'time_peak_gust': df.loc[gusts.idxmax(), 'valid'] if len(gusts) > 0 else pd.NaT,
            
            'mean_wind': winds.mean() if len(winds) > 0 else np.nan,
            'max_1hr_mean_wind': winds.rolling(12, min_periods=6).mean().max() if len(winds) > 0 else np.nan,  # 12 obs = 1 hr at 5-min
            
            'pct_missing_wind': (df['sknt'].isna().sum() / len(df)) * 100,
            'pct_missing_gust': (df['gust'].isna().sum() / len(df)) * 100,
        }
        
        return result
    
    def download_all_stations(self):
        """Download and process data for all stations."""
        results = []
        
        for station in self.stations:
            df = self.download_station_data(station)
            processed = self.process_station_data(df, station)
            
            if processed is not None:
                results.append(processed)
        
        return pd.DataFrame(results)
    
    def save_data(self, df):
        """Save processed ASOS data to CSV."""
        output_file = self.output_dir / f'{self.event}_asos_summary.csv'
        df.to_csv(output_file, index=False)
        print("Saved:", output_file)
        
        meta_file = self.output_dir / f'{self.event}_metadata.json'
        metadata = {
            'event': self.event,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'n_stations': len(df),
            'stations': df['station'].tolist(),
            'downloaded': datetime.now().isoformat()
        }
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Saved:", meta_file)
        
        return output_file
    
    def generate_summary_stats(self, df):
        """Print summary statistics."""

        peak = df['peak_gust'].max()
        mean = df['peak_gust'].mean()
        median = df['peak_gust'].median()

def main():
    parser = argparse.ArgumentParser(description='Download and process ASOS data for WRF comparison')
    parser.add_argument('--event', required=True, 
                       choices=['houston', 'iowa', 'june2012', 'marshall'],
                       help='Event name')
    parser.add_argument('--start', required=True,
                       help='Start time (UTC) in format "YYYY-MM-DD HH:MM"')
    parser.add_argument('--end', required=True,
                       help='End time (UTC) in format "YYYY-MM-DD HH:MM"')
    parser.add_argument('--output-dir', default='data/asos',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    downloader = ASOSDownloader(
        event=args.event,
        start_time=args.start,
        end_time=args.end,
        output_dir=args.output_dir
    )
    
    df = downloader.download_all_stations()
    
    if df.empty:
        print("ERROR: No data downloaded successfully")
        return 1
    
    downloader.save_data(df)
    downloader.generate_summary_stats(df)
    
    return 0


if __name__ == '__main__':
    exit(main())
