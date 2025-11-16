import os
import sys
import json
import time
import getpass
import webbrowser
import threading
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
# FIX: Import timezone to handle timezone-aware datetimes
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from requests.auth import HTTPBasicAuth

# --- CONSTANTS FOR UNIT CONVERSION ---
MILES_PER_METER = 0.000621371
KM_PER_METER = 0.001
MPH_PER_MPS = 2.23694
KPH_PER_MPS = 3.6


class StravaAuthenticator:
    """
    Handles the browser-based authentication flow with the Strava API.
    """
    
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None
        self.client_id = None
        self.client_secret = None
        self.auth_code = None

    def authenticate(self) -> bool:
        """Authenticate with Strava API using a local callback server."""
        print("\n" + "="*50)
        print("STRAVA API AUTHENTICATION")
        print("="*50)
        print("\nFor security, please enter your Strava API credentials.")
        print("Your credentials will not be stored permanently.\n")
        
        self.client_id = getpass.getpass("Enter your Strava Client ID: ")
        self.client_secret = getpass.getpass("Enter your Strava Client Secret: ")

        port = 53682
        redirect_uri = f"http://localhost:{port}/callback"

        class _CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                authenticator = self.server.authenticator
                query_components = parse_qs(urlparse(self.path).query)
                if 'code' in query_components:
                    authenticator.auth_code = query_components["code"][0]
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"<html><head><title>Authentication Success</title></head>")
                    self.wfile.write(b"<body style='font-family: sans-serif; text-align: center; padding-top: 50px;'>")
                    self.wfile.write(b"<h1>Authentication Successful!</h1>")
                    self.wfile.write(b"<p>You can now close this browser window and return to the application.</p>")
                    self.wfile.write(b"</body></html>")
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b"<h1>Authentication Failed</h1><p>No authorization code found.</p>")
                threading.Thread(target=self.server.shutdown).start()

        auth_url = (f"https://www.strava.com/oauth/authorize?"
                    f"client_id={self.client_id}&response_type=code&"
                    f"redirect_uri={redirect_uri}&scope=activity:read_all")

        with HTTPServer(('localhost', port), _CallbackHandler) as server:
            server.authenticator = self
            print("\nOpening your browser to authorize this application...")
            webbrowser.open(auth_url)
            server.serve_forever()

        if not self.auth_code:
            print("\n✗ Could not capture authorization code. Authentication failed.")
            return False
            
        print("\n✓ Authorization code received. Exchanging for access token...")

        token_url = "https://www.strava.com/oauth/token"
        payload = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': self.auth_code,
            'grant_type': 'authorization_code'
        }
        
        try:
            response = requests.post(token_url, data=payload)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data['refresh_token']
                self.expires_at = token_data['expires_at']
                print("\n✓ Authentication successful!")
                return True
            else:
                print(f"\n✗ Authentication failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"\n✗ Authentication error: {e}")
            return False
    
    def refresh_access_token(self) -> bool:
        """Refresh the access token if expired"""
        if self.expires_at and time.time() > self.expires_at:
            token_url = "https://www.strava.com/oauth/token"
            payload = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'refresh_token': self.refresh_token,
                'grant_type': 'refresh_token'
            }
            response = requests.post(token_url, data=payload)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data['refresh_token']
                self.expires_at = token_data['expires_at']
        return True


class StravaDataFetcher:
    """Fetches data from Strava API for a specified time period."""
    def __init__(self, authenticator: StravaAuthenticator, unit_system: str):
        self.auth = authenticator
        self.unit_system = unit_system
        self.base_url = "https://www.strava.com/api/v3"
        
    def fetch_athlete_info(self) -> Dict:
        self.auth.refresh_access_token()
        headers = {'Authorization': f'Bearer {self.auth.access_token}'}
        response = requests.get(f"{self.base_url}/athlete", headers=headers)
        return response.json() if response.status_code == 200 else {}
    
    def fetch_activities(self, period: timedelta) -> pd.DataFrame:
        self.auth.refresh_access_token()
        headers = {'Authorization': f'Bearer {self.auth.access_token}'}
        
        # FIX: Use timezone-aware datetime for accurate API calls across timezones
        end_date = datetime.now(timezone.utc)
        start_date = end_date - period
        
        activities, page, per_page = [], 1, 100
        print(f"\nFetching activities from the last {period.days} days...")
        
        while True:
            params = {'after': int(start_date.timestamp()), 'before': int(end_date.timestamp()), 'page': page, 'per_page': per_page}
            response = requests.get(f"{self.base_url}/athlete/activities", headers=headers, params=params)
            
            if response.status_code != 200: break
            batch = response.json()
            if not batch: break
            
            runs = [a for a in batch if a.get('type') == 'Run']
            activities.extend(runs)
            print(f"  Fetched page {page}: {len(runs)} runs")
            page += 1
            if len(batch) < per_page: break
        
        print(f"✓ Total runs fetched: {len(activities)}")
        
        if activities:
            df = pd.DataFrame(activities)
            # This line correctly creates a timezone-aware DatetimeIndex (UTC)
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['moving_time_minutes'] = df['moving_time'] / 60
            
            if self.unit_system == 'km':
                df['distance_unit'] = df['distance'] * KM_PER_METER
            else:
                df['distance_unit'] = df['distance'] * MILES_PER_METER

            df['pace_per_unit'] = df['moving_time_minutes'] / df['distance_unit']
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df
        
        return pd.DataFrame()


class MarathonTrainingAnalyzer:
    """Core analysis engine with dynamic time aggregation and corrected predictions."""
    
    def __init__(self, activities_df: pd.DataFrame, unit_system: str):
        self.df = activities_df
        self.unit_system = unit_system
        self.agg_stats = None
        self.long_runs = None
        self.agg_period = 'W' 
        self.agg_label = 'Weekly'
        self._stats_calculated = False

    def calculate_aggregate_stats(self) -> pd.DataFrame:
        if self._stats_calculated:
            return self.agg_stats

        if self.df.empty: return pd.DataFrame()
        
        duration_days = (self.df['start_date'].max() - self.df['start_date'].min()).days
        if duration_days > 180:
            self.agg_period = 'M'
            self.agg_label = 'Monthly'
            print("\nData spans over 6 months. Aggregating by month for clarity.")
        else:
            self.agg_period = 'W'
            self.agg_label = 'Weekly'
            print("\nAggregating by week.")

        agg_col = self.df['start_date'].dt.to_period(self.agg_period)
        
        aggregated = self.df.groupby(agg_col).agg({
            'distance_unit': 'sum', 'moving_time_minutes': 'sum', 'id': 'count',
            'average_heartrate': 'mean', 'pace_per_unit': 'mean'
        }).round(2)
        
        aggregated.columns = ['total_distance', 'total_minutes', 'run_count', 'avg_hr', 'avg_pace']
        aggregated['avg_run_distance'] = aggregated['total_distance'] / aggregated['run_count']
        aggregated['distance_change'] = aggregated['total_distance'].pct_change() * 100
        
        self.agg_stats = aggregated
        self._stats_calculated = True
        return aggregated
    
    def analyze_long_runs(self, min_distance: Optional[float] = None) -> pd.DataFrame:
        if min_distance is None:
            min_distance = 16.0 if self.unit_system == 'km' else 10.0

        long_runs = self.df[self.df['distance_unit'] >= min_distance].copy()
        if not long_runs.empty:
            long_runs = long_runs.sort_values('start_date')
            long_runs['days_since_last'] = long_runs['start_date'].diff().dt.days
            self.long_runs = long_runs
            return long_runs
        return pd.DataFrame()
    
    def predict_marathon_time(self) -> Dict:
        """Predicts marathon time based on recent performance (last 6 months)."""
        # FIX: Create a timezone-aware cutoff date to compare against the tz-aware 'start_date' column.
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)
        recent_df = self.df[self.df['start_date'] > cutoff_date]
        
        if recent_df.empty:
            return {}

        predictions = {}
        if self.unit_system == 'km':
            race_distances = {'5K': (4.8, 5.2), '10K': (9.8, 10.2), 'Half Marathon': (20.9, 21.3)}
            marathon_dist = 42.195
        else:
            race_distances = {'5K': (3.0, 3.3), '10K': (6.0, 6.5), 'Half Marathon': (13.0, 13.3)}
            marathon_dist = 26.2

        for race_name, (min_dist, max_dist) in race_distances.items():
            races = recent_df[(recent_df['distance_unit'] >= min_dist) & (recent_df['distance_unit'] <= max_dist)]
            if not races.empty:
                best_race = races.nsmallest(1, 'pace_per_unit').iloc[0]
                pace = best_race['pace_per_unit']
                
                if race_name == '5K': marathon_pace = pace * 1.15
                elif race_name == '10K': marathon_pace = pace * 1.11
                else: marathon_pace = pace * 1.06
                
                predictions[race_name] = {
                    'predicted_time': marathon_pace * marathon_dist,
                    'predicted_pace': marathon_pace,
                    'based_on_pace': pace
                }
        
        if predictions:
            avg_time = np.mean([p['predicted_time'] for p in predictions.values()])
            predictions['average'] = {'predicted_time': avg_time, 'predicted_pace': avg_time / marathon_dist}
        
        return predictions


class TrainingVisualizer:
    """Creates visualizations with corrected dynamic aggregation logic."""
    
    def __init__(self, analyzer: MarathonTrainingAnalyzer):
        self.analyzer = analyzer
        self.unit = self.analyzer.unit_system
        
    def create_training_dashboard(self) -> go.Figure:
        """Fetches the aggregation label at runtime for correct plot titles."""
        agg_stats = self.analyzer.calculate_aggregate_stats()
        agg_label = self.analyzer.agg_label

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(f'{agg_label} Distance ({self.unit})', f'Long Run Progression ({self.unit})',
                            'Pace Distribution', f'{agg_label} Run Count'),
        )
        if not agg_stats.empty:
            fig.add_trace(go.Scatter(x=agg_stats.index.astype(str), y=agg_stats['total_distance'], mode='lines+markers'), row=1, col=1)
            fig.add_trace(go.Bar(x=agg_stats.index.astype(str), y=agg_stats['run_count']), row=2, col=2)

        long_runs = self.analyzer.analyze_long_runs()
        if not long_runs.empty:
            fig.add_trace(go.Scatter(x=long_runs['start_date'], y=long_runs['distance_unit'], mode='lines+markers'), row=1, col=2)
        
        if not self.analyzer.df.empty:
            fig.add_trace(go.Histogram(x=self.analyzer.df['pace_per_unit']), row=2, col=1)

        fig.update_layout(height=800, title_text="Training Analysis Dashboard", showlegend=False)
        return fig
    
    def plot_progression(self) -> plt.Figure:
        agg_stats = self.analyzer.calculate_aggregate_stats()
        agg_label = self.analyzer.agg_label
        
        if agg_stats is None or agg_stats.empty: return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{agg_label} Training Progression', fontsize=16)

        axes[0, 0].plot(agg_stats.index.astype(str), agg_stats['total_distance'], marker='o')
        axes[0, 0].set_title(f'{agg_label} Distance Progression')
        axes[0, 0].set_ylabel(f'Distance ({self.unit})')
        axes[0, 1].bar(agg_stats.index.astype(str), agg_stats['run_count'], color='skyblue')
        axes[0, 1].set_title(f'{agg_label} Run Frequency')
        axes[1, 0].plot(agg_stats.index.astype(str), agg_stats['avg_pace'], marker='s', color='green')
        axes[1, 0].set_title(f'Average {agg_label} Pace')
        axes[1, 0].set_ylabel(f'Minutes per {self.unit}')
        colors = ['red' if x < 0 else 'green' for x in agg_stats['distance_change'].fillna(0)]
        axes[1, 1].bar(agg_stats.index.astype(str), agg_stats['distance_change'].fillna(0), color=colors)
        axes[1, 1].set_title(f'Period-over-Period Distance Change (%)')

        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig


class TrainingReportGenerator:
    """Generates and prints reports with fully implemented logic."""
    
    def __init__(self, analyzer: MarathonTrainingAnalyzer):
        self.analyzer = analyzer
        self.unit = self.analyzer.unit_system
        
    def generate_training_report(self) -> Dict:
        report = {'summary': {}, 'aggregate_analysis': {}, 'long_run_analysis': {}, 'predictions': {}}
        
        if not self.analyzer.df.empty:
            duration_weeks = max(1, (self.analyzer.df['start_date'].max() - self.analyzer.df['start_date'].min()).days / 7)
            report['summary'] = {
                'total_runs': len(self.analyzer.df),
                f'total_distance_{self.unit}': round(self.analyzer.df['distance_unit'].sum(), 1),
                f'avg_weekly_distance_{self.unit}': round(self.analyzer.df['distance_unit'].sum() / duration_weeks, 1),
                f'longest_run_{self.unit}': round(self.analyzer.df['distance_unit'].max(), 1),
            }
        
        agg_stats = self.analyzer.calculate_aggregate_stats()
        if not agg_stats.empty:
            report['aggregate_analysis'] = {
                f'peak_{self.analyzer.agg_label}_distance_{self.unit}': round(agg_stats['total_distance'].max(), 1),
                'consistency_score (%)': round(100 * (1 - agg_stats['total_distance'].std() / agg_stats['total_distance'].mean()), 1),
                f'avg_runs_per_{self.analyzer.agg_label[:-2].lower()}': round(agg_stats['run_count'].mean(), 1)
            }
        
        long_runs = self.analyzer.analyze_long_runs()
        if not long_runs.empty:
            report['long_run_analysis'] = {
                'total_long_runs': len(long_runs),
                f'longest_run_{self.unit}': round(long_runs['distance_unit'].max(), 1),
            }
        
        report['predictions'] = self.analyzer.predict_marathon_time()
        return report

    def print_report(self, report: Dict, sections_to_print: Optional[List[str]] = None):
        if not sections_to_print:
            sections_to_print = ['summary', 'aggregate_analysis', 'long_run_analysis', 'predictions']

        print("\n" + "="*60)
        print("TRAINING REPORT")
        print("="*60)
        
        def format_key(key):
            return key.replace(f'_{self.unit}', '').replace('_', ' ').title()

        for section in sections_to_print:
            if report.get(section):
                title = section.replace('_', ' ').upper()
                print(f"\n-- {title} --")
                print("-"*40)
                
                if section == 'predictions':
                    if 'average' in report[section]:
                        avg_pred = report[section]['average']
                        t = avg_pred['predicted_time']
                        h, m, s = int(t // 60), int(t % 60), int((t * 60) % 60)
                        print(f"  Predicted Finish Time: {h}:{m:02d}:{s:02d}")
                        p = avg_pred['predicted_pace']
                        p_min, p_sec = int(p), int((p * 60) % 60)
                        print(f"  Target Pace: {p_min}:{p_sec:02d} per {self.unit}")
                    else:
                        print("  Not enough recent race data to make a prediction.")
                else:
                    for key, value in report[section].items():
                        print(f"  {format_key(key)}: {value}")
        
        print("\n" + "="*60)


class MarathonTrainingApp:
    """Main application with fixed menu options."""
    
    def __init__(self):
        self.authenticator = StravaAuthenticator()
        self.unit_system = None
        self.fetcher = None
        self.analyzer = None
        self.visualizer = None
        self.reporter = None
        
    def run(self):
        print("\n" + "="*60)
        print("MARATHON TRAINING ANALYZER FOR STRAVA")
        print("="*60)
        
        while self.unit_system not in ['km', 'miles']:
            choice = input("\nSelect your preferred unit system (km/miles): ").strip().lower()
            if choice in ['km', 'miles']: self.unit_system = choice
            else: print("❌ Invalid input. Please enter 'km' or 'miles'.")
        print(f"✓ Units set to {self.unit_system}.")
        
        if not self.authenticator.authenticate(): return
        
        self.fetcher = StravaDataFetcher(self.authenticator, self.unit_system)
        athlete = self.fetcher.fetch_athlete_info()
        if athlete: print(f"\nHello, {athlete.get('firstname', 'Athlete')}!")
        
        while True:
            print("\n" + "-"*40 + "\nMAIN MENU\n" + "-"*40)
            print("1. Fetch/Refresh Activity Data\n2. View Training Summary")
            print("3. Analyze Progression\n4. Analyze Long Runs")
            print("5. View Marathon Predictions\n6. Generate Full Report")
            print("7. Create Training Dashboard\n8. Export Data\n9. Exit")
            choice = input("\nSelect an option (1-9): ").strip()
            
            if choice == '1': self._fetch_data()
            elif choice == '2': self._view_summary()
            elif choice == '3': self._analyze_progression()
            elif choice == '4': self._analyze_long_runs()
            elif choice == '5': self._view_predictions()
            elif choice == '6': self._generate_report()
            elif choice == '7': self._create_dashboard()
            elif choice == '8': self._export_data()
            elif choice == '9':
                print("\nThank you for using the Marathon Training Analyzer!")
                break
            else:
                print("\n❌ Invalid option. Please try again.")

    def _fetch_data(self):
        while True:
            period_str = input("\nEnter time period to fetch (e.g., 16w, 6m, 2y) [default: 12w]: ").strip()
            if not period_str: period_str = '12w'
            
            period = self._parse_time_period(period_str)
            if period: break
            else: print("❌ Invalid format. Use 'w' (weeks), 'm' (months), or 'y' (years).")

        df = self.fetcher.fetch_activities(period)
        if not df.empty:
            self.analyzer = MarathonTrainingAnalyzer(df, self.unit_system)
            self.visualizer = TrainingVisualizer(self.analyzer)
            self.reporter = TrainingReportGenerator(self.analyzer)
            print(f"\n✓ Successfully loaded {len(df)} activities")
        else:
            print("\nNo activities found in the specified time range.")
    
    def _view_summary(self):
        if not self._check_data(): return
        report = self.reporter.generate_training_report()
        self.reporter.print_report(report, sections_to_print=['summary'])
    
    def _analyze_progression(self):
        if not self._check_data(): return
        agg_stats = self.analyzer.calculate_aggregate_stats()
        print(f"\n-- {self.analyzer.agg_label.upper()} PROGRESSION --")
        print(agg_stats.to_string())
        if fig := self.visualizer.plot_progression(): plt.show()
    
    def _view_predictions(self):
        if not self._check_data(): return
        print("\nNote: Predictions are based on your best race efforts in the last 6 months.")
        report = self.reporter.generate_training_report()
        self.reporter.print_report(report, sections_to_print=['predictions'])
        
    def _generate_report(self):
        if not self._check_data(): return
        report = self.reporter.generate_training_report()
        self.reporter.print_report(report)
        if input("\nSave report to file? (y/n): ").strip().lower() == 'y':
            filename = f"marathon_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f: json.dump(report, f, indent=2, default=str)
            print(f"✓ Report saved to {filename}")

    def _create_dashboard(self):
        if not self._check_data(): return
        print("\nGenerating interactive dashboard...")
        fig = self.visualizer.create_training_dashboard()
        filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        print(f"✓ Dashboard saved to {filename}")
        try:
            webbrowser.open(f"file://{os.path.abspath(filename)}")
        except Exception:
            print(f"Please open {filename} in your web browser.")
            
    def _parse_time_period(self, period_str: str) -> Optional[timedelta]:
        match = re.match(r"(\d+)([wmy])", period_str.lower())
        if not match: return None
        value, unit = int(match.group(1)), match.group(2)
        if unit == 'w': return timedelta(weeks=value)
        if unit == 'm': return timedelta(days=value * 30)
        if unit == 'y': return timedelta(days=value * 365)
        return None

    def _analyze_long_runs(self):
        if not self._check_data(): return
        default_dist = 16 if self.unit_system == 'km' else 10
        min_dist_str = input(f"\nMinimum distance for long runs (default: {default_dist} {self.unit_system}): ").strip()
        min_dist = float(min_dist_str) if min_dist_str and min_dist_str.replace('.', '', 1).isdigit() else default_dist
        
        long_runs = self.analyzer.analyze_long_runs(min_dist)
        if not long_runs.empty:
            print(f"\n-- LONG RUNS (>= {min_dist} {self.unit_system}) --")
            print(long_runs[['start_date', 'distance_unit', 'pace_per_unit', 'days_since_last']].to_string())
        else:
            print(f"\nNo runs found >= {min_dist} {self.unit_system}.")

    def _export_data(self):
        if not self._check_data(): return
        filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.analyzer.df.to_csv(filename, index=False)
        print(f"✓ Data exported to {filename}")
    
    def _check_data(self) -> bool:
        if self.analyzer is None or self.analyzer.df.empty:
            print("\nNo data loaded. Please fetch data first (Option 1).")
            return False
        return True

def main():
    app = MarathonTrainingApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()