#!/usr/bin/env python3
"""
analyze_wrf_winds.py
--------------------
Extract WRF 10m winds at ASOS locations using distance-weighted averaging,
compare to observations, and train ML models for bias correction.

Example use:
    python analyze_wrf_winds.py --event houston --wrfout wrfout_d02_* --asos data/asos/houston_asos_summary.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.spatial.distance import cdist
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between points.
    
    Parameters
    ----------
    lat1, lon1 : float or array
        First point(s) latitude and longitude in degrees
    lat2, lon2 : float or array
        Second point(s) latitude and longitude in degrees
    
    Returns
    -------
    distance : float or array
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


class WRFWindExtractor:
    """Extract WRF 10m winds at ASOS station locations."""
    
    def __init__(self, wrfout_files, radius_km=5):
        """
        Initialize WRF wind extraction fucntion.
        
        Parameters
        ----------
        wrfout_files : list of str
            List of wrfout file paths
        radius_km : float
            Radius in km for weighted averaging (default is 15)
        """
        self.wrfout_files = sorted(wrfout_files)
        self.radius_km = radius_km
        self._load_grid()
    
    def _load_grid(self):
        """Load WRF grid coordinates from first file."""
        
        with Dataset(self.wrfout_files[0], 'r') as nc:
            self.wrf_lat = nc.variables['XLAT'][0, :, :]
            self.wrf_lon = nc.variables['XLONG'][0, :, :]
            self.map_proj = nc.MAP_PROJ
            self.dx = nc.DX
            self.dy = nc.DY
            
        self.nlat, self.nlon = self.wrf_lat.shape
    
    def extract_at_station(self, station_lat, station_lon, wrf_field):
        """
        Extract WRF value at station location using distance-weighted averaging.
        
        Parameters
        ----------
        station_lat, station_lon : float
            Station coordinates
        wrf_field : 2D array
            WRF field to extract
        
        Returns
        -------
        dict
            Statistics: weighted_mean, max, min, std, n_points
        """
        wrf_lat_flat = self.wrf_lat.flatten()
        wrf_lon_flat = self.wrf_lon.flatten()
        wrf_field_flat = wrf_field.flatten()
        
        # Calculate distances from station to all grid points
        distances = haversine_distance(
            station_lat, station_lon,
            wrf_lat_flat, wrf_lon_flat
        )
        
        mask = distances <= self.radius_km
        
        if not np.any(mask):
            # Fallback to nearest point
            nearest_idx = np.argmin(distances)
            return {
                'weighted_mean': wrf_field_flat[nearest_idx],
                'max': wrf_field_flat[nearest_idx],
                'min': wrf_field_flat[nearest_idx],
                'std': 0.0,
                'n_points': 1,
                'min_distance': distances[nearest_idx]
            }
        
        # Calculate inverse distance squared weights
        distances_subset = distances[mask]
        weights = 1.0 / (distances_subset**2 + 0.01)  # Add small value to avoid dividing by zero
        weights = weights / np.sum(weights)  # Normalize
        
        values = wrf_field_flat[mask]
        
        return {
            'weighted_mean': np.sum(values * weights),
            'max': np.max(values),
            'min': np.min(values),
            'std': np.std(values),
            'n_points': np.sum(mask),
            'min_distance': np.min(distances_subset)
        }
    
    def extract_all_stations(self, asos_df, time_idx=None):
        """
        Extract WRF winds at all ASOS stations for a given time.
        
        Parameters
        ----------
        asos_df : DataFrame
            ASOS station metadata with 'lat', 'lon', 'station' columns
        time_idx : int, optional
            Time index in WRF file. If None, computes runtime maximum.
        
        Returns
        -------
        DataFrame
            WRF winds at each station
        """
        results = []
        
        # Determine which files and times to process
        if time_idx is None:
            # Compare peak gusts (default)
            print(f"Extracting runtime maximum winds from {len(self.wrfout_files)} files...")
            compute_max = True
        else:
            # Can also do specific time index if we want
            print(f"Extracting winds at time index {time_idx}...")
            compute_max = False
        
        for idx, row in asos_df.iterrows():
            station = row['station']
            lat = row['lat']
            lon = row['lon']
            
            # Accumulate max winds across all files
            if compute_max:
                u10_max = -999
                v10_max = -999
                wspd_max = -999
                
                for wrfout_file in self.wrfout_files:
                    with Dataset(wrfout_file, 'r') as nc:
                        ntimes = nc.dimensions['Time'].size
                        
                        for t in range(ntimes):
                            # Get 10m winds (U10, V10 in m/s)
                            u10 = nc.variables['U10'][t, :, :]
                            v10 = nc.variables['V10'][t, :, :]
                            wspd = np.sqrt(u10**2 + v10**2)
                            
                            # Extract at station
                            u_stats = self.extract_at_station(lat, lon, u10)
                            v_stats = self.extract_at_station(lat, lon, v10)
                            wspd_stats = self.extract_at_station(lat, lon, wspd)
                            
                            # Update max
                            if wspd_stats['weighted_mean'] > wspd_max:
                                wspd_max = wspd_stats['weighted_mean']
                                u10_max = u_stats['weighted_mean']
                                v10_max = v_stats['weighted_mean']
                
                result = {
                    'station': station,
                    'lat': lat,
                    'lon': lon,
                    'wrf_u10_max': u10_max,
                    'wrf_v10_max': v10_max,
                    'wrf_wspd_max': wspd_max,
                }
            
            else:
                # Extract at specific time
                with Dataset(self.wrfout_files[0], 'r') as nc:
                    u10 = nc.variables['U10'][time_idx, :, :]
                    v10 = nc.variables['V10'][time_idx, :, :]
                    wspd = np.sqrt(u10**2 + v10**2)
                
                u_stats = self.extract_at_station(lat, lon, u10)
                v_stats = self.extract_at_station(lat, lon, v10)
                wspd_stats = self.extract_at_station(lat, lon, wspd)
                
                result = {
                    'station': station,
                    'lat': lat,
                    'lon': lon,
                    'wrf_u10': u_stats['weighted_mean'],
                    'wrf_v10': v_stats['weighted_mean'],
                    'wrf_wspd': wspd_stats['weighted_mean'],
                    'wrf_wspd_max_neighborhood': wspd_stats['max'],
                    'n_grid_points': wspd_stats['n_points'],
                    'min_distance_km': wspd_stats['min_distance']
                }
            
            results.append(result)
            
            if (idx + 1) % 5 == 0:
                print(f"  Processed {idx + 1}/{len(asos_df)} stations")
        
        return pd.DataFrame(results)
    


### Machine Learning Section

class WindMLAnalysis:
    """Machine learning analysis of WRF vs ASOS winds."""
    
    def __init__(self, merged_df, event_name):
        """
        Initialize ML analysis function.
        
        Parameters
        ----------
        merged_df : DataFrame
            Merged WRF and ASOS data
        event_name : str
            Event identifier for grouping
        """
        self.df = merged_df
        self.event = event_name
        self._compute_features()
            
    def _compute_features(self):
        """Compute WRF bias."""
        self.df['bias'] = self.df['wrf_wspd_max'] - self.df['obs_gust']
        self.df['bias_pct'] = (self.df['bias'] / self.df['obs_gust']) * 100
    
    def train_models(self):
        """Train multiple ML models for bias correction."""
        feature_cols = ['wrf_wspd_max', 'lat', 'lon', 'elevation']

        X = self.df[feature_cols].values
        y = self.df['obs_gust'].values  # Predict observed gust
        
        # Train several models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(name)
            
            model.fit(X, y)
            y_pred = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            corrected_bias = y_pred - y
            mean_corrected_bias = np.mean(corrected_bias)
            
            print("RMSE:", rmse, "m/s")
            print("MAE:", mae, "m/s")
            print("R^2:", r2)
            print("Residual bias:", mean_corrected_bias, "m/s")
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'features': feature_cols
            }
        
        self.models = results
        return results
    
    def generate_comparison_plots(self, output_dir='figures'):
        """Generate comparison plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        
        # Figure 1: Scatter plot - WRF vs Observations
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: Raw WRF
        ax = axes[0]
        ax.scatter(self.df['obs_gust'], self.df['wrf_wspd_max'], 
                  alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        
        max_val = max(self.df['obs_gust'].max(), self.df['wrf_wspd_max'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
        
        rmse_raw = np.sqrt(mean_squared_error(self.df['obs_gust'], self.df['wrf_wspd_max']))
        bias_raw = self.df['bias'].mean()
        
        ax.text(0.05, 0.95, 
               f'RMSE = {rmse_raw:.2f} m/s\nBias = {bias_raw:.2f} m/s',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Observed Wind Gust (m/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('WRF 10-m Wind Max (m/s)', fontsize=12, fontweight='bold')
        ax.set_title('(a) Raw WRF vs Observations', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel B: ML-corrected WRF (best model)
        if hasattr(self, 'models'):
            best_model_name = min(self.models.keys(), key=lambda k: self.models[k]['rmse'])
            predictions = self.models[best_model_name]['predictions']
            
            ax = axes[1]
            ax.scatter(self.df['obs_gust'], predictions,
                      alpha=0.6, s=60, edgecolors='black', linewidth=0.5,
                      color='green')
            
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1 line')
            
            rmse_ml = self.models[best_model_name]['rmse']
            bias_ml = np.mean(predictions - self.df['obs_gust'])
            
            ax.text(0.05, 0.95,
                   f'{best_model_name}\nRMSE = {rmse_ml:.2f} m/s\nBias = {bias_ml:.2f} m/s',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            ax.set_xlabel('Observed Wind Gust (m/s)', fontsize=12, fontweight='bold')
            ax.set_ylabel('ML-Corrected Wind (m/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'(b) {best_model_name} Correction', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_file = output_dir / f'{self.event}_scatter_comparison.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Bias distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(self.df['bias'], bins=20, alpha=0.6, color='red', 
               edgecolor='black', label='Raw WRF bias')
        
        if hasattr(self, 'models'):
            ml_bias = predictions - self.df['obs_gust']
            ax.hist(ml_bias, bins=20, alpha=0.6, color='green',
                   edgecolor='black', label=f'{best_model_name} bias')
        
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Bias (WRF - Observed) [m/s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Stations', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.event.upper()}: Wind Prediction Bias Distribution', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig_file = output_dir / f'{self.event}_bias_histogram.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Spatial map of bias
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        scatter = ax.scatter(self.df['lon'], self.df['lat'],
                            c=self.df['bias'], s=100,
                            cmap='RdBu_r', vmin=-15, vmax=15,
                            edgecolors='black', linewidth=0.5)
        
        # Add station labels for top biases
        for _, row in self.df.nlargest(3, 'bias').iterrows():
            ax.annotate(row['station'], 
                       (row['lon'], row['lat']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, label='Bias (m/s)')
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.event.upper()}: Spatial Distribution of WRF Wind Bias',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_file = output_dir / f'{self.event}_bias_map.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze WRF winds vs ASOS observations')
    parser.add_argument('--event', required=True, help='Event name')
    parser.add_argument('--wrfout', nargs='+', required=True, 
                       help='WRF output files (wrfout_d02_*)')
    parser.add_argument('--asos', required=True, 
                       help='ASOS summary CSV file')
    parser.add_argument('--radius', type=float, default=5,
                       help='Weighted averaging radius in km (default: 5)')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    asos_df = pd.read_csv(args.asos)

    extractor = WRFWindExtractor(args.wrfout, radius_km=args.radius)
    wrf_df = extractor.extract_all_stations(asos_df)
  
    merged_df = asos_df.merge(wrf_df, on='station', suffixes=('', '_wrf'))
    merged_df = merged_df.rename(columns={'peak_gust': 'obs_gust'})
    
    merged_file = output_dir / f'{args.event}_merged_data.csv'
    merged_df.to_csv(merged_file, index=False)
    
    ml_analysis = WindMLAnalysis(merged_df, args.event)
    ml_analysis.train_models()
    ml_analysis.generate_comparison_plots(output_dir / 'figures')
    
    return 0


if __name__ == '__main__':
    exit(main())
