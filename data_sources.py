"""
Data source abstractions for loading running activity data from various sources.
Supports Strava API, CSV files, Excel files, and cached data.
"""

import os
import json
import time
import pickle
import getpass
import webbrowser
import threading
from abc import ABC, abstractmethod
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import chardet

# --- CONSTANTS FOR UNIT CONVERSION ---
MILES_PER_METER = 0.000621371
KM_PER_METER = 0.001
MPH_PER_MPS = 2.23694
KPH_PER_MPS = 3.6


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, unit_system: str):
        """
        Initialize data source with unit system.

        Args:
            unit_system: Either 'km' or 'miles'
        """
        self.unit_system = unit_system

    @abstractmethod
    def load_activities(self, **kwargs) -> pd.DataFrame:
        """
        Load activities and return a standardized DataFrame.

        Returns:
            DataFrame with standardized columns:
                - activity_id: Unique identifier
                - start_date: Timezone-aware datetime
                - name: Activity name
                - distance_meters: Distance in meters
                - moving_time_seconds: Moving time
                - elapsed_time_seconds: Total elapsed time
                - distance_unit: Distance in preferred unit
                - moving_time_minutes: Moving time in minutes
                - pace_per_unit: Minutes per unit distance
                - average_hr: Average heart rate (optional)
                - max_hr: Max heart rate (optional)
                - elevation_gain: Elevation gain in meters (optional)
                - calories: Calories burned (optional)
                - relative_effort: Training load metric (optional)
        """
        pass

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard transformations to the DataFrame.

        Args:
            df: Raw DataFrame with at least distance_meters and moving_time_seconds

        Returns:
            DataFrame with calculated fields
        """
        if df.empty:
            return df

        # Ensure start_date is timezone-aware
        if 'start_date' in df.columns:
            if df['start_date'].dt.tz is None:
                df['start_date'] = df['start_date'].dt.tz_localize('UTC')

        # Calculate derived fields
        df['moving_time_minutes'] = df['moving_time_seconds'] / 60

        if self.unit_system == 'km':
            df['distance_unit'] = df['distance_meters'] * KM_PER_METER
        else:
            df['distance_unit'] = df['distance_meters'] * MILES_PER_METER

        # Calculate pace (minutes per unit distance)
        df['pace_per_unit'] = df['moving_time_minutes'] / df['distance_unit']

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df


class StravaAuthenticator:
    """Handles the browser-based authentication flow with the Strava API."""

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
            def log_message(self, format, *args):
                pass  # Suppress server logs

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
        """Refresh the access token if expired."""
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


class StravaAPISource(DataSource):
    """Fetches data from Strava API for a specified time period."""

    def __init__(self, authenticator: StravaAuthenticator, unit_system: str):
        super().__init__(unit_system)
        self.auth = authenticator
        self.base_url = "https://www.strava.com/api/v3"

    def fetch_athlete_info(self) -> Dict:
        """Fetch athlete information from Strava API."""
        self.auth.refresh_access_token()
        headers = {'Authorization': f'Bearer {self.auth.access_token}'}
        response = requests.get(f"{self.base_url}/athlete", headers=headers)
        return response.json() if response.status_code == 200 else {}

    def load_activities(self, period: timedelta = timedelta(weeks=12)) -> pd.DataFrame:
        """
        Fetch activities from Strava API.

        Args:
            period: Time period to fetch (default 12 weeks)

        Returns:
            Standardized DataFrame of running activities
        """
        self.auth.refresh_access_token()
        headers = {'Authorization': f'Bearer {self.auth.access_token}'}

        end_date = datetime.now(timezone.utc)
        start_date = end_date - period

        activities, page, per_page = [], 1, 100
        print(f"\nFetching activities from the last {period.days} days...")

        while True:
            params = {
                'after': int(start_date.timestamp()),
                'before': int(end_date.timestamp()),
                'page': page,
                'per_page': per_page
            }
            response = requests.get(f"{self.base_url}/athlete/activities", headers=headers, params=params)

            if response.status_code != 200:
                break
            batch = response.json()
            if not batch:
                break

            runs = [a for a in batch if a.get('type') == 'Run']
            activities.extend(runs)
            print(f"  Fetched page {page}: {len(runs)} runs")
            page += 1
            if len(batch) < per_page:
                break

        print(f"✓ Total runs fetched: {len(activities)}")

        if not activities:
            return pd.DataFrame()

        # Convert to standardized format
        df = pd.DataFrame(activities)

        standardized = pd.DataFrame({
            'activity_id': df['id'].astype(str),
            'start_date': pd.to_datetime(df['start_date']),
            'name': df['name'],
            'distance_meters': df['distance'],
            'moving_time_seconds': df['moving_time'],
            'elapsed_time_seconds': df['elapsed_time'],
            'average_hr': df.get('average_heartrate'),
            'max_hr': df.get('max_heartrate'),
            'elevation_gain': df.get('total_elevation_gain'),
            'calories': df.get('calories'),
            'average_speed': df.get('average_speed'),
        })

        return self._standardize_dataframe(standardized)


class CSVFileSource(DataSource):
    """Loads activity data from Strava CSV export files."""

    STRAVA_DATE_FORMATS = [
        '%d %b %Y, %H:%M:%S',  # "20 Aug 2014, 17:38:32"
        '%Y-%m-%d %H:%M:%S',   # "2014-08-20 17:38:32"
        '%m/%d/%Y %H:%M:%S',   # "08/20/2014 17:38:32"
        '%d/%m/%Y %H:%M:%S',   # "20/08/2014 17:38:32"
    ]

    def __init__(self, unit_system: str):
        super().__init__(unit_system)

    def load_activities(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load activities from a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Standardized DataFrame of running activities

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        print(f"\nLoading CSV file: {file_path}")

        # Try multiple encodings
        encodings_to_try = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252']

        # Detect file encoding
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(50000)
                detected = chardet.detect(raw_data)
                detected_encoding = detected['encoding']
                if detected_encoding and detected_encoding not in encodings_to_try:
                    encodings_to_try.insert(0, detected_encoding)
        except Exception:
            pass

        df = None
        last_error = None

        # Try each encoding
        for encoding in encodings_to_try:
            try:
                print(f"Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"✓ Successfully read CSV with {encoding} encoding")
                break
            except Exception as e:
                last_error = e
                continue

        if df is None:
            raise ValueError(f"Failed to parse CSV file with any encoding. Last error: {last_error}")

        print(f"✓ Loaded {len(df)} total activities")

        # Filter for runs only
        if 'Activity Type' in df.columns:
            df = df[df['Activity Type'] == 'Run'].copy()
            print(f"✓ Filtered to {len(df)} running activities")
        elif 'Type' in df.columns:
            df = df[df['Type'] == 'Run'].copy()
            print(f"✓ Filtered to {len(df)} running activities")

        if df.empty:
            print("✗ No running activities found in CSV")
            return pd.DataFrame()

        # Parse the CSV into standardized format
        return self._parse_strava_csv(df)

    def _parse_strava_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse Strava CSV format into standardized format.

        Args:
            df: Raw DataFrame from CSV

        Returns:
            Standardized DataFrame
        """
        # Map column names (handle variations)
        col_map = {
            'Activity ID': 'activity_id',
            'Activity Date': 'start_date',
            'Start Time': 'start_date_alt',
            'Activity Name': 'name',
            'Distance': 'distance',  # This is the second Distance column (index 17)
            'Moving Time': 'moving_time',
            'Elapsed Time': 'elapsed_time',
            'Average Heart Rate': 'average_hr',
            'Max Heart Rate': 'max_hr',
            'Elevation Gain': 'elevation_gain',
            'Calories': 'calories',
            'Relative Effort': 'relative_effort',
            'Average Speed': 'average_speed',
        }

        # Create standardized dataframe
        standardized = pd.DataFrame()

        # Activity ID
        if 'Activity ID' in df.columns:
            standardized['activity_id'] = df['Activity ID'].astype(str)
        else:
            # Generate IDs if not present
            standardized['activity_id'] = ['CSV_' + str(i) for i in range(len(df))]

        # Date parsing
        date_col = 'Activity Date' if 'Activity Date' in df.columns else 'Start Time'
        if date_col in df.columns:
            standardized['start_date'] = self._parse_dates(df[date_col])
        else:
            raise ValueError("No date column found in CSV")

        # Name
        if 'Activity Name' in df.columns:
            standardized['name'] = df['Activity Name']
        else:
            standardized['name'] = 'Run'

        # Distance (meters) - Use the detailed Distance column (index 17 in Strava exports)
        # In pandas, duplicate column names get .1, .2 suffixes
        distance_col = None
        if 'Distance' in df.columns:
            # If there are duplicate Distance columns, we want the second one
            distance_cols = [col for col in df.columns if col == 'Distance' or col.startswith('Distance.')]
            if len(distance_cols) > 1:
                distance_col = distance_cols[1]  # Use second Distance column
            else:
                distance_col = 'Distance'

        if distance_col and distance_col in df.columns:
            standardized['distance_meters'] = pd.to_numeric(df[distance_col], errors='coerce')
        else:
            raise ValueError("No distance column found in CSV")

        # Moving Time (seconds)
        if 'Moving Time' in df.columns:
            standardized['moving_time_seconds'] = pd.to_numeric(df['Moving Time'], errors='coerce')
        elif 'Elapsed Time' in df.columns:
            standardized['moving_time_seconds'] = pd.to_numeric(df['Elapsed Time'], errors='coerce')
        else:
            raise ValueError("No time column found in CSV")

        # Elapsed Time (seconds)
        elapsed_cols = [col for col in df.columns if col == 'Elapsed Time' or col.startswith('Elapsed Time.')]
        if len(elapsed_cols) > 1:
            elapsed_col = elapsed_cols[1]  # Use second Elapsed Time column
        elif 'Elapsed Time' in df.columns:
            elapsed_col = 'Elapsed Time'
        else:
            elapsed_col = None

        if elapsed_col:
            standardized['elapsed_time_seconds'] = pd.to_numeric(df[elapsed_col], errors='coerce')
        else:
            standardized['elapsed_time_seconds'] = standardized['moving_time_seconds']

        # Optional fields
        if 'Average Heart Rate' in df.columns:
            standardized['average_hr'] = pd.to_numeric(df['Average Heart Rate'], errors='coerce')

        if 'Max Heart Rate' in df.columns:
            standardized['max_hr'] = pd.to_numeric(df['Max Heart Rate'], errors='coerce')

        if 'Elevation Gain' in df.columns:
            standardized['elevation_gain'] = pd.to_numeric(df['Elevation Gain'], errors='coerce')

        if 'Calories' in df.columns:
            standardized['calories'] = pd.to_numeric(df['Calories'], errors='coerce')

        if 'Relative Effort' in df.columns:
            # Use the second Relative Effort if there are duplicates
            re_cols = [col for col in df.columns if col == 'Relative Effort' or col.startswith('Relative Effort.')]
            re_col = re_cols[1] if len(re_cols) > 1 else 'Relative Effort'
            standardized['relative_effort'] = pd.to_numeric(df[re_col], errors='coerce')

        if 'Average Speed' in df.columns:
            standardized['average_speed'] = pd.to_numeric(df['Average Speed'], errors='coerce')

        # Remove rows with missing critical data
        standardized = standardized.dropna(subset=['start_date', 'distance_meters', 'moving_time_seconds'])
        standardized = standardized[standardized['distance_meters'] > 0]
        standardized = standardized[standardized['moving_time_seconds'] > 0]

        print(f"✓ Successfully parsed {len(standardized)} valid activities")

        return self._standardize_dataframe(standardized)

    def _parse_dates(self, date_series: pd.Series) -> pd.Series:
        """
        Parse dates from various formats.

        Args:
            date_series: Series of date strings

        Returns:
            Series of timezone-aware datetime objects
        """
        parsed_dates = None

        for date_format in self.STRAVA_DATE_FORMATS:
            try:
                parsed_dates = pd.to_datetime(date_series, format=date_format, errors='coerce')
                valid_count = parsed_dates.notna().sum()
                if valid_count > len(date_series) * 0.9:  # If >90% parsed successfully
                    break
            except Exception:
                continue

        # If no format worked, try pandas auto-detection
        if parsed_dates is None or parsed_dates.isna().all():
            parsed_dates = pd.to_datetime(date_series, errors='coerce')

        # Make timezone-aware (assume UTC if not specified)
        if parsed_dates.dt.tz is None:
            parsed_dates = parsed_dates.dt.tz_localize('UTC')

        return parsed_dates


class ExcelFileSource(DataSource):
    """Loads activity data from Excel files (.xlsx, .xls)."""

    def __init__(self, unit_system: str):
        super().__init__(unit_system)
        self.csv_source = CSVFileSource(unit_system)

    def load_activities(self, file_path: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load activities from an Excel file.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of sheet to load (default: first sheet)

        Returns:
            Standardized DataFrame of running activities

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        print(f"\nLoading Excel file: {file_path}")

        try:
            # Try to load Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path, sheet_name=0)  # First sheet
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")

        print(f"✓ Loaded {len(df)} total activities from Excel")

        # Filter for runs
        if 'Activity Type' in df.columns:
            df = df[df['Activity Type'] == 'Run'].copy()
            print(f"✓ Filtered to {len(df)} running activities")
        elif 'Type' in df.columns:
            df = df[df['Type'] == 'Run'].copy()
            print(f"✓ Filtered to {len(df)} running activities")

        if df.empty:
            print("✗ No running activities found in Excel file")
            return pd.DataFrame()

        # Use CSV parser logic for standardization
        return self.csv_source._parse_strava_csv(df)


class CachedDataSource(DataSource):
    """Loads and saves cached activity data locally."""

    CACHE_DIR = Path.home() / '.marathon_analyzer' / 'cache'

    def __init__(self, unit_system: str):
        super().__init__(unit_system)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load_activities(self, cache_name: str = 'default', **kwargs) -> pd.DataFrame:
        """
        Load activities from cache.

        Args:
            cache_name: Name of the cache file

        Returns:
            Standardized DataFrame of activities
        """
        cache_file = self.CACHE_DIR / f"{cache_name}.pkl"

        if not cache_file.exists():
            print(f"\n✗ No cached data found: {cache_name}")
            return pd.DataFrame()

        print(f"\nLoading cached data: {cache_name}")

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            df = data['dataframe']
            metadata = data.get('metadata', {})

            print(f"✓ Loaded {len(df)} activities from cache")
            print(f"  Cached on: {metadata.get('cached_at', 'Unknown')}")
            print(f"  Source: {metadata.get('source', 'Unknown')}")

            return df
        except Exception as e:
            print(f"✗ Failed to load cache: {e}")
            return pd.DataFrame()

    def save_cache(self, df: pd.DataFrame, cache_name: str = 'default', source: str = 'Unknown'):
        """
        Save activities to cache.

        Args:
            df: DataFrame to cache
            cache_name: Name for the cache file
            source: Description of data source
        """
        cache_file = self.CACHE_DIR / f"{cache_name}.pkl"

        data = {
            'dataframe': df,
            'metadata': {
                'cached_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': source,
                'unit_system': self.unit_system,
                'row_count': len(df),
            }
        }

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"\n✓ Cached {len(df)} activities as '{cache_name}'")
        except Exception as e:
            print(f"\n✗ Failed to save cache: {e}")

    def list_caches(self) -> List[Tuple[str, dict]]:
        """
        List all available caches.

        Returns:
            List of (cache_name, metadata) tuples
        """
        caches = []
        for cache_file in self.CACHE_DIR.glob('*.pkl'):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                cache_name = cache_file.stem
                caches.append((cache_name, data.get('metadata', {})))
            except Exception:
                continue
        return caches


def merge_activities(dfs: List[pd.DataFrame], deduplicate: bool = True) -> pd.DataFrame:
    """
    Merge multiple activity DataFrames and optionally deduplicate.

    Args:
        dfs: List of DataFrames to merge
        deduplicate: Whether to remove duplicate activities

    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()

    # Filter out empty dataframes
    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        return pd.DataFrame()

    # Concatenate all dataframes
    merged = pd.concat(dfs, ignore_index=True)

    if deduplicate:
        print(f"\nMerging {len(merged)} total activities...")

        # Strategy: Remove duplicates based on date + distance + time
        # (activity_id might differ between sources)
        merged['_dedup_key'] = (
            merged['start_date'].dt.strftime('%Y-%m-%d %H:%M') + '_' +
            merged['distance_meters'].round(0).astype(str) + '_' +
            merged['moving_time_seconds'].round(0).astype(str)
        )

        before_count = len(merged)
        merged = merged.drop_duplicates(subset=['_dedup_key'], keep='first')
        merged = merged.drop(columns=['_dedup_key'])
        after_count = len(merged)

        duplicates_removed = before_count - after_count
        if duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate activities")

        print(f"✓ Final dataset: {after_count} unique activities")

    # Sort by date
    merged = merged.sort_values('start_date').reset_index(drop=True)

    return merged
