"""
Advanced analysis engines for marathon training data.
Includes training load, predictions, heart rate zones, and more.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')


class MarathonTrainingAnalyzer:
    """Core analysis engine with dynamic time aggregation and comprehensive metrics."""

    def __init__(self, activities_df: pd.DataFrame, unit_system: str):
        """
        Initialize analyzer with activity data.

        Args:
            activities_df: DataFrame of activities in standardized format
            unit_system: Either 'km' or 'miles'
        """
        self.df = activities_df.copy() if not activities_df.empty else pd.DataFrame()
        self.unit_system = unit_system
        self.agg_stats = None
        self.long_runs = None
        self.agg_period = 'W'
        self.agg_label = 'Weekly'
        self._stats_calculated = False

        # Heart rate zones (will be set by user or default)
        self.hr_zones = None

    def calculate_aggregate_stats(self) -> pd.DataFrame:
        """
        Calculate weekly or monthly aggregate statistics.

        Returns:
            DataFrame with aggregated statistics per period
        """
        if self._stats_calculated and self.agg_stats is not None:
            return self.agg_stats

        if self.df.empty:
            return pd.DataFrame()

        duration_days = (self.df['start_date'].max() - self.df['start_date'].min()).days

        # Dynamic aggregation: monthly if >6 months, otherwise weekly
        if duration_days > 180:
            self.agg_period = 'M'
            self.agg_label = 'Monthly'
            print("\nData spans over 6 months. Aggregating by month for clarity.")
        else:
            self.agg_period = 'W'
            self.agg_label = 'Weekly'
            print("\nAggregating by week.")

        agg_col = self.df['start_date'].dt.to_period(self.agg_period)

        agg_dict = {
            'distance_unit': 'sum',
            'moving_time_minutes': 'sum',
            'activity_id': 'count',
        }

        # Add optional aggregations if data exists
        if 'average_hr' in self.df.columns and self.df['average_hr'].notna().any():
            agg_dict['average_hr'] = 'mean'

        if 'pace_per_unit' in self.df.columns:
            agg_dict['pace_per_unit'] = 'mean'

        if 'elevation_gain' in self.df.columns and self.df['elevation_gain'].notna().any():
            agg_dict['elevation_gain'] = 'sum'

        if 'relative_effort' in self.df.columns and self.df['relative_effort'].notna().any():
            agg_dict['relative_effort'] = 'sum'

        aggregated = self.df.groupby(agg_col).agg(agg_dict).round(2)

        # Rename columns
        col_rename = {
            'distance_unit': 'total_distance',
            'moving_time_minutes': 'total_minutes',
            'activity_id': 'run_count',
            'average_hr': 'avg_hr',
            'pace_per_unit': 'avg_pace',
            'elevation_gain': 'total_elevation',
            'relative_effort': 'total_effort',
        }
        aggregated.rename(columns=col_rename, inplace=True)

        # Calculate derived metrics
        aggregated['avg_run_distance'] = (aggregated['total_distance'] / aggregated['run_count']).round(2)
        aggregated['distance_change'] = (aggregated['total_distance'].pct_change() * 100).round(1)

        self.agg_stats = aggregated
        self._stats_calculated = True
        return aggregated

    def analyze_long_runs(self, min_distance: Optional[float] = None) -> pd.DataFrame:
        """
        Analyze long runs above a minimum distance threshold.

        Args:
            min_distance: Minimum distance to qualify as long run (default: 16km or 10mi)

        Returns:
            DataFrame of long runs with analysis
        """
        if min_distance is None:
            min_distance = 16.0 if self.unit_system == 'km' else 10.0

        long_runs = self.df[self.df['distance_unit'] >= min_distance].copy()

        if not long_runs.empty:
            long_runs = long_runs.sort_values('start_date')
            long_runs['days_since_last'] = long_runs['start_date'].diff().dt.days
            self.long_runs = long_runs
            return long_runs

        return pd.DataFrame()

    def calculate_training_load(self) -> pd.DataFrame:
        """
        Calculate training load metrics (ATL, CTL, TSB).

        ATL (Acute Training Load): 7-day rolling average of training stress
        CTL (Chronic Training Load): 42-day rolling average (fitness)
        TSB (Training Stress Balance): CTL - ATL (freshness)

        Returns:
            DataFrame with daily training load metrics
        """
        if self.df.empty:
            return pd.DataFrame()

        # Create daily aggregation
        daily = self.df.groupby(self.df['start_date'].dt.date).agg({
            'distance_unit': 'sum',
            'moving_time_minutes': 'sum',
            'relative_effort': 'sum' if 'relative_effort' in self.df.columns else 'count',
        }).reset_index()

        daily.columns = ['date', 'distance', 'time', 'effort']

        # If no relative_effort, use distance as proxy for training stress
        if 'relative_effort' not in self.df.columns or daily['effort'].isna().all():
            daily['training_stress'] = daily['distance'] * 1.0
        else:
            daily['training_stress'] = daily['effort']

        # Calculate rolling averages
        daily['ATL'] = daily['training_stress'].rolling(window=7, min_periods=1).mean()  # Acute (7-day)
        daily['CTL'] = daily['training_stress'].rolling(window=42, min_periods=1).mean()  # Chronic (42-day)
        daily['TSB'] = daily['CTL'] - daily['ATL']  # Training Stress Balance

        return daily

    def detect_injury_risks(self) -> Dict[str, any]:
        """
        Detect potential injury risks based on training patterns.

        Returns:
            Dictionary with risk warnings and metrics
        """
        risks = {
            'rapid_increase': [],
            'insufficient_recovery': [],
            'high_monotony': [],
            'warnings': []
        }

        if self.df.empty:
            return risks

        # Calculate weekly distances
        weekly = self.df.groupby(self.df['start_date'].dt.to_period('W')).agg({
            'distance_unit': 'sum'
        })

        # Check for rapid week-over-week increases (>10% is risky)
        weekly['pct_change'] = weekly['distance_unit'].pct_change() * 100

        for period, row in weekly.iterrows():
            if row['pct_change'] > 10:
                risks['rapid_increase'].append({
                    'period': str(period),
                    'increase_pct': round(row['pct_change'], 1),
                    'distance': round(row['distance_unit'], 1)
                })
                risks['warnings'].append(
                    f"⚠ Week {period}: {row['pct_change']:.1f}% increase in mileage (>10% is risky)"
                )

        # Check for insufficient recovery (3+ days running in a row without rest)
        self.df_sorted = self.df.sort_values('start_date')
        consecutive_days = 0
        last_date = None

        for idx, row in self.df_sorted.iterrows():
            current_date = row['start_date'].date()
            if last_date and (current_date - last_date).days == 1:
                consecutive_days += 1
                if consecutive_days >= 3:
                    risks['insufficient_recovery'].append({
                        'date': str(current_date),
                        'consecutive_days': consecutive_days + 1
                    })
            else:
                consecutive_days = 0
            last_date = current_date

        if risks['insufficient_recovery']:
            risks['warnings'].append(
                f"⚠ Found {len(risks['insufficient_recovery'])} instances of 4+ consecutive running days"
            )

        # Calculate training monotony (weekly)
        for period, group in self.df.groupby(self.df['start_date'].dt.to_period('W')):
            if len(group) >= 3:
                distances = group['distance_unit'].values
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                monotony = mean_dist / std_dist if std_dist > 0 else 0

                if monotony > 2.0:  # High monotony is risky
                    risks['high_monotony'].append({
                        'period': str(period),
                        'monotony': round(monotony, 2)
                    })

        if risks['high_monotony']:
            risks['warnings'].append(
                f"⚠ Found {len(risks['high_monotony'])} weeks with high training monotony (>2.0)"
            )

        return risks

    def analyze_heart_rate_zones(self, max_hr: Optional[int] = None,
                                 resting_hr: Optional[int] = None) -> Dict:
        """
        Analyze time spent in different heart rate zones.

        Args:
            max_hr: Maximum heart rate (default: estimate from data)
            resting_hr: Resting heart rate (optional, for HRR calculation)

        Returns:
            Dictionary with zone analysis
        """
        if self.df.empty or 'average_hr' not in self.df.columns:
            return {}

        hr_data = self.df[self.df['average_hr'].notna()].copy()

        if hr_data.empty:
            return {}

        # Estimate max HR if not provided
        if max_hr is None:
            max_hr = int(hr_data['max_hr'].max()) if 'max_hr' in hr_data.columns else 190

        # Define zones (% of max HR)
        zones = {
            'Zone 1 (Recovery)': (0.50, 0.60),
            'Zone 2 (Aerobic)': (0.60, 0.70),
            'Zone 3 (Tempo)': (0.70, 0.80),
            'Zone 4 (Threshold)': (0.80, 0.90),
            'Zone 5 (VO2 Max)': (0.90, 1.00),
        }

        self.hr_zones = zones
        zone_analysis = {}

        for zone_name, (lower, upper) in zones.items():
            lower_hr = max_hr * lower
            upper_hr = max_hr * upper

            zone_runs = hr_data[
                (hr_data['average_hr'] >= lower_hr) &
                (hr_data['average_hr'] < upper_hr)
            ]

            zone_analysis[zone_name] = {
                'run_count': len(zone_runs),
                'total_time_minutes': zone_runs['moving_time_minutes'].sum(),
                'total_distance': zone_runs['distance_unit'].sum(),
                'pct_of_runs': (len(zone_runs) / len(hr_data) * 100) if len(hr_data) > 0 else 0,
                'hr_range': f"{int(lower_hr)}-{int(upper_hr)} bpm"
            }

        zone_analysis['max_hr_used'] = max_hr
        zone_analysis['total_runs_with_hr'] = len(hr_data)

        return zone_analysis

    def predict_marathon_time(self) -> Dict:
        """
        Predict marathon time using multiple models.

        Returns:
            Dictionary with predictions from different models
        """
        # Use recent data (last 6 months)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)
        recent_df = self.df[self.df['start_date'] > cutoff_date]

        if recent_df.empty:
            return {}

        predictions = {}
        marathon_dist = 42.195 if self.unit_system == 'km' else 26.2

        # Define race distance ranges
        if self.unit_system == 'km':
            race_distances = {
                '5K': (4.8, 5.2),
                '10K': (9.8, 10.2),
                'Half Marathon': (20.9, 21.3)
            }
        else:
            race_distances = {
                '5K': (3.0, 3.3),
                '10K': (6.0, 6.5),
                'Half Marathon': (13.0, 13.3)
            }

        # Model 1: Riegel Formula
        for race_name, (min_dist, max_dist) in race_distances.items():
            races = recent_df[
                (recent_df['distance_unit'] >= min_dist) &
                (recent_df['distance_unit'] <= max_dist)
            ]

            if not races.empty:
                best_race = races.nsmallest(1, 'pace_per_unit').iloc[0]
                pace = best_race['pace_per_unit']
                race_dist = best_race['distance_unit']

                # Riegel formula with adjusted fatigue factors
                if race_name == '5K':
                    marathon_pace = pace * 1.15
                elif race_name == '10K':
                    marathon_pace = pace * 1.11
                else:  # Half marathon
                    marathon_pace = pace * 1.06

                predictions[f'Riegel_{race_name}'] = {
                    'predicted_time_minutes': marathon_pace * marathon_dist,
                    'predicted_pace': marathon_pace,
                    'based_on_pace': pace,
                    'model': 'Riegel',
                    'source_race': race_name
                }

        # Model 2: VDOT-based prediction (Daniels)
        for race_name, (min_dist, max_dist) in race_distances.items():
            races = recent_df[
                (recent_df['distance_unit'] >= min_dist) &
                (recent_df['distance_unit'] <= max_dist)
            ]

            if not races.empty:
                best_race = races.nsmallest(1, 'pace_per_unit').iloc[0]
                race_time = best_race['moving_time_minutes']
                race_dist_km = best_race['distance_unit'] if self.unit_system == 'km' else best_race['distance_unit'] * 1.60934

                # Simplified VDOT calculation
                vdot = self._calculate_vdot(race_dist_km, race_time)
                marathon_time = self._vdot_to_marathon_time(vdot)

                if marathon_time:
                    predictions[f'VDOT_{race_name}'] = {
                        'predicted_time_minutes': marathon_time,
                        'predicted_pace': marathon_time / marathon_dist,
                        'vdot': vdot,
                        'model': 'VDOT (Daniels)',
                        'source_race': race_name
                    }

        # Model 3: Cameron Formula (distance-based)
        best_long_run = recent_df.nlargest(1, 'distance_unit')
        if not best_long_run.empty:
            longest = best_long_run.iloc[0]
            long_run_pace = longest['pace_per_unit']

            # Cameron: Marathon pace ≈ long run pace + 5%
            cameron_pace = long_run_pace * 1.05

            predictions['Cameron_LongRun'] = {
                'predicted_time_minutes': cameron_pace * marathon_dist,
                'predicted_pace': cameron_pace,
                'based_on_distance': longest['distance_unit'],
                'model': 'Cameron',
                'source_race': 'Longest Run'
            }

        # Average prediction
        if predictions:
            avg_time = np.mean([p['predicted_time_minutes'] for p in predictions.values()])
            min_time = np.min([p['predicted_time_minutes'] for p in predictions.values()])
            max_time = np.max([p['predicted_time_minutes'] for p in predictions.values()])

            predictions['Average'] = {
                'predicted_time_minutes': avg_time,
                'predicted_pace': avg_time / marathon_dist,
                'min_time': min_time,
                'max_time': max_time,
                'model': 'Average of all models',
                'confidence_range_minutes': max_time - min_time
            }

        return predictions

    def _calculate_vdot(self, distance_km: float, time_minutes: float) -> float:
        """
        Calculate VDOT (VO2 Max estimate) from race performance.

        Args:
            distance_km: Distance in kilometers
            time_minutes: Time in minutes

        Returns:
            VDOT value
        """
        # Simplified VDOT formula (Jack Daniels)
        velocity_m_min = (distance_km * 1000) / time_minutes  # meters per minute
        pct_vo2max = 0.8 + 0.1894393 * np.exp(-0.012778 * time_minutes) + 0.2989558 * np.exp(-0.1932605 * time_minutes)

        vo2 = -4.60 + 0.182258 * velocity_m_min + 0.000104 * velocity_m_min ** 2
        vdot = vo2 / pct_vo2max

        return max(30, min(85, vdot))  # Clamp to reasonable range

    def _vdot_to_marathon_time(self, vdot: float) -> Optional[float]:
        """
        Convert VDOT to predicted marathon time.

        Args:
            vdot: VDOT value

        Returns:
            Predicted marathon time in minutes
        """
        # Marathon distance in meters
        marathon_m = 42195

        # Calculate marathon velocity from VDOT
        # Using Daniels' formula inverted
        time_minutes_estimate = marathon_m / (vdot * 0.98 * 0.182258)  # Simplified

        # More accurate iterative approach
        for time_est in range(120, 420, 1):  # 2-7 hours
            velocity = marathon_m / time_est
            pct_vo2max = 0.8 + 0.1894393 * np.exp(-0.012778 * time_est) + 0.2989558 * np.exp(-0.1932605 * time_est)
            vo2 = -4.60 + 0.182258 * velocity + 0.000104 * velocity ** 2
            calc_vdot = vo2 / pct_vo2max

            if calc_vdot <= vdot:
                return float(time_est)

        return time_minutes_estimate

    def compare_periods(self, period1_start: datetime, period1_end: datetime,
                       period2_start: datetime, period2_end: datetime) -> Dict:
        """
        Compare training between two time periods.

        Args:
            period1_start: Start of first period
            period1_end: End of first period
            period2_start: Start of second period
            period2_end: End of second period

        Returns:
            Dictionary with comparison metrics
        """
        if self.df.empty:
            return {}

        # Ensure timezone-aware
        if period1_start.tzinfo is None:
            period1_start = period1_start.replace(tzinfo=timezone.utc)
        if period1_end.tzinfo is None:
            period1_end = period1_end.replace(tzinfo=timezone.utc)
        if period2_start.tzinfo is None:
            period2_start = period2_start.replace(tzinfo=timezone.utc)
        if period2_end.tzinfo is None:
            period2_end = period2_end.replace(tzinfo=timezone.utc)

        period1 = self.df[
            (self.df['start_date'] >= period1_start) &
            (self.df['start_date'] <= period1_end)
        ]

        period2 = self.df[
            (self.df['start_date'] >= period2_start) &
            (self.df['start_date'] <= period2_end)
        ]

        def calc_stats(df):
            if df.empty:
                return {}
            return {
                'run_count': len(df),
                'total_distance': df['distance_unit'].sum(),
                'avg_distance': df['distance_unit'].mean(),
                'total_time_hours': df['moving_time_minutes'].sum() / 60,
                'avg_pace': df['pace_per_unit'].mean(),
                'longest_run': df['distance_unit'].max(),
            }

        p1_stats = calc_stats(period1)
        p2_stats = calc_stats(period2)

        comparison = {
            'period1': {
                'start': period1_start.strftime('%Y-%m-%d'),
                'end': period1_end.strftime('%Y-%m-%d'),
                'stats': p1_stats
            },
            'period2': {
                'start': period2_start.strftime('%Y-%m-%d'),
                'end': period2_end.strftime('%Y-%m-%d'),
                'stats': p2_stats
            },
            'changes': {}
        }

        # Calculate changes
        if p1_stats and p2_stats:
            for key in p1_stats:
                if key in p2_stats and p1_stats[key] != 0:
                    change_pct = ((p2_stats[key] - p1_stats[key]) / p1_stats[key]) * 100
                    comparison['changes'][key] = {
                        'absolute': p2_stats[key] - p1_stats[key],
                        'percent': change_pct
                    }

        return comparison

    def get_training_summary(self) -> Dict:
        """
        Get comprehensive training summary.

        Returns:
            Dictionary with all summary statistics
        """
        if self.df.empty:
            return {}

        duration_weeks = max(1, (self.df['start_date'].max() - self.df['start_date'].min()).days / 7)

        summary = {
            'total_runs': len(self.df),
            f'total_distance_{self.unit_system}': round(self.df['distance_unit'].sum(), 1),
            f'avg_weekly_distance_{self.unit_system}': round(self.df['distance_unit'].sum() / duration_weeks, 1),
            f'longest_run_{self.unit_system}': round(self.df['distance_unit'].max(), 1),
            f'avg_run_distance_{self.unit_system}': round(self.df['distance_unit'].mean(), 1),
            'avg_pace_per_unit': round(self.df['pace_per_unit'].mean(), 2),
            'total_time_hours': round(self.df['moving_time_minutes'].sum() / 60, 1),
            'date_range': f"{self.df['start_date'].min().strftime('%Y-%m-%d')} to {self.df['start_date'].max().strftime('%Y-%m-%d')}"
        }

        # Add optional metrics if available
        if 'average_hr' in self.df.columns and self.df['average_hr'].notna().any():
            summary['avg_heart_rate'] = round(self.df['average_hr'].mean(), 0)

        if 'elevation_gain' in self.df.columns and self.df['elevation_gain'].notna().any():
            summary['total_elevation_gain_m'] = round(self.df['elevation_gain'].sum(), 0)

        if 'calories' in self.df.columns and self.df['calories'].notna().any():
            summary['total_calories'] = round(self.df['calories'].sum(), 0)

        return summary
