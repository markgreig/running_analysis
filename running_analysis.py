#!/usr/bin/env python3
"""
Marathon Training Analyzer - Enhanced Version
Supports Strava API, CSV/Excel imports, advanced analytics, and more.
"""

import os
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings('ignore')

# Import modules
from data_sources import (
    StravaAuthenticator, StravaAPISource, CSVFileSource,
    ExcelFileSource, CachedDataSource, merge_activities
)
from analyzers import MarathonTrainingAnalyzer
from visualizers import TrainingVisualizer
from reporters import TrainingReportGenerator
from config import Config


class MarathonTrainingApp:
    """Main application with enhanced features and file import support."""

    def __init__(self):
        """Initialize the application."""
        self.config = Config()
        self.authenticator = None
        self.unit_system = None
        self.analyzer = None
        self.visualizer = None
        self.reporter = None
        self.data_loaded = False
        self.current_data_source = None

    def run(self):
        """Run the main application loop."""
        print("\n" + "="*70)
        print(" " * 15 + "MARATHON TRAINING ANALYZER")
        print(" " * 22 + "Enhanced Version")
        print("="*70)

        # Check if first run
        if not self.config.CONFIG_FILE.exists():
            print("\nWelcome! It looks like this is your first time running the app.")
            run_setup = input("Would you like to run the setup wizard? (y/n): ").strip().lower()
            if run_setup == 'y':
                self.config.setup_wizard()

        # Load unit system from config
        self.unit_system = self.config.get_unit_system()
        print(f"\n✓ Unit system set to: {self.unit_system}")

        # Main menu loop
        while True:
            self._show_main_menu()
            choice = input("\nSelect an option (1-15): ").strip()

            if choice == '1':
                self._load_data_menu()
            elif choice == '2':
                self._view_summary()
            elif choice == '3':
                self._analyze_progression()
            elif choice == '4':
                self._analyze_long_runs()
            elif choice == '5':
                self._view_predictions()
            elif choice == '6':
                self._analyze_training_load()
            elif choice == '7':
                self._analyze_heart_rate_zones()
            elif choice == '8':
                self._check_injury_risks()
            elif choice == '9':
                self._compare_periods()
            elif choice == '10':
                self._generate_full_report()
            elif choice == '11':
                self._create_visualizations()
            elif choice == '12':
                self._manage_goals()
            elif choice == '13':
                self._export_data()
            elif choice == '14':
                self._settings_menu()
            elif choice == '15':
                print("\n✓ Thank you for using Marathon Training Analyzer!")
                break
            else:
                print("\n✗ Invalid option. Please try again.")

    def _show_main_menu(self):
        """Display the main menu."""
        print("\n" + "-"*70)
        print("MAIN MENU")
        print("-"*70)
        print(" 1. Load Data (API / CSV / Excel / Cache)")
        print(" 2. View Training Summary")
        print(" 3. Analyze Progression")
        print(" 4. Analyze Long Runs")
        print(" 5. View Marathon Predictions")
        print(" 6. Analyze Training Load (ATL/CTL/TSB)")
        print(" 7. Analyze Heart Rate Zones")
        print(" 8. Check Injury Risks")
        print(" 9. Compare Training Periods")
        print("10. Generate Full Report")
        print("11. Create Visualizations")
        print("12. Manage Goals")
        print("13. Export Data")
        print("14. Settings")
        print("15. Exit")

        if self.data_loaded:
            print(f"\n[Data loaded: {len(self.analyzer.df)} activities from {self.current_data_source}]")

    def _load_data_menu(self):
        """Show data loading submenu."""
        print("\n" + "-"*70)
        print("LOAD DATA")
        print("-"*70)
        print("1. Fetch from Strava API")
        print("2. Import from CSV file")
        print("3. Import from Excel file")
        print("4. Load cached data")
        print("5. Merge multiple sources")
        print("6. Back to main menu")

        choice = input("\nSelect data source (1-6): ").strip()

        if choice == '1':
            self._load_from_strava_api()
        elif choice == '2':
            self._load_from_csv()
        elif choice == '3':
            self._load_from_excel()
        elif choice == '4':
            self._load_from_cache()
        elif choice == '5':
            self._merge_multiple_sources()
        elif choice == '6':
            return
        else:
            print("\n✗ Invalid option")

    def _load_from_strava_api(self):
        """Load data from Strava API."""
        if not self.authenticator:
            self.authenticator = StravaAuthenticator()
            if not self.authenticator.authenticate():
                return

        source = StravaAPISource(self.authenticator, self.unit_system)

        # Get athlete info
        athlete = source.fetch_athlete_info()
        if athlete:
            print(f"\nHello, {athlete.get('firstname', 'Athlete')}!")

        # Get time period
        default_period = self.config.get('default_fetch_period', '12w')
        period_str = input(f"\nEnter time period to fetch (e.g., 16w, 6m, 2y) [default: {default_period}]: ").strip()
        if not period_str:
            period_str = default_period

        period = self._parse_time_period(period_str)
        if not period:
            print("\n✗ Invalid time period format")
            return

        # Fetch activities
        df = source.load_activities(period=period)

        if not df.empty:
            self._initialize_analyzer(df, "Strava API")

            # Offer to cache
            if self.config.get('cache_enabled', True):
                cache_it = input("\nSave to cache for faster loading next time? (y/n): ").strip().lower()
                if cache_it == 'y':
                    cache_source = CachedDataSource(self.unit_system)
                    cache_source.save_cache(df, cache_name='strava_latest', source='Strava API')
        else:
            print("\n✗ No activities found")

    def _load_from_csv(self):
        """Load data from CSV file."""
        file_path = input("\nEnter CSV file path: ").strip()

        # Remove quotes if present
        file_path = file_path.strip('"').strip("'")

        source = CSVFileSource(self.unit_system)

        try:
            df = source.load_activities(file_path)
            if not df.empty:
                self._initialize_analyzer(df, f"CSV: {Path(file_path).name}")

                # Offer to cache
                if self.config.get('cache_enabled', True):
                    cache_it = input("\nSave to cache? (y/n): ").strip().lower()
                    if cache_it == 'y':
                        cache_source = CachedDataSource(self.unit_system)
                        cache_name = input("Cache name [csv_import]: ").strip() or 'csv_import'
                        cache_source.save_cache(df, cache_name=cache_name, source=f"CSV: {file_path}")
            else:
                print("\n✗ No activities loaded from CSV")
        except Exception as e:
            print(f"\n✗ Error loading CSV: {e}")

    def _load_from_excel(self):
        """Load data from Excel file."""
        file_path = input("\nEnter Excel file path: ").strip()
        file_path = file_path.strip('"').strip("'")

        source = ExcelFileSource(self.unit_system)

        try:
            df = source.load_activities(file_path)
            if not df.empty:
                self._initialize_analyzer(df, f"Excel: {Path(file_path).name}")

                # Offer to cache
                if self.config.get('cache_enabled', True):
                    cache_it = input("\nSave to cache? (y/n): ").strip().lower()
                    if cache_it == 'y':
                        cache_source = CachedDataSource(self.unit_system)
                        cache_name = input("Cache name [excel_import]: ").strip() or 'excel_import'
                        cache_source.save_cache(df, cache_name=cache_name, source=f"Excel: {file_path}")
            else:
                print("\n✗ No activities loaded from Excel")
        except Exception as e:
            print(f"\n✗ Error loading Excel: {e}")

    def _load_from_cache(self):
        """Load data from cache."""
        cache_source = CachedDataSource(self.unit_system)
        caches = cache_source.list_caches()

        if not caches:
            print("\n✗ No cached data found")
            return

        print("\nAvailable caches:")
        for i, (cache_name, metadata) in enumerate(caches, 1):
            print(f"{i}. {cache_name} ({metadata.get('row_count', '?')} activities, cached: {metadata.get('cached_at', 'Unknown')})")

        choice = input(f"\nSelect cache (1-{len(caches)}) or 'c' to cancel: ").strip()

        if choice.lower() == 'c':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(caches):
                cache_name = caches[idx][0]
                df = cache_source.load_activities(cache_name=cache_name)
                if not df.empty:
                    self._initialize_analyzer(df, f"Cache: {cache_name}")
            else:
                print("\n✗ Invalid selection")
        except ValueError:
            print("\n✗ Invalid input")

    def _merge_multiple_sources(self):
        """Merge data from multiple sources."""
        print("\nMerge multiple data sources (duplicates will be removed)")
        print("Load sources one by one. Enter 'done' when finished.\n")

        dfs = []
        source_names = []

        while True:
            print("\nCurrent sources loaded:", len(dfs))
            print("1. Add Strava API data")
            print("2. Add CSV file")
            print("3. Add Excel file")
            print("4. Add cached data")
            print("5. Done - merge all sources")

            choice = input("\nSelect option: ").strip()

            if choice == '5':
                break
            elif choice in ['1', '2', '3', '4']:
                # Temporarily load data
                temp_analyzer = self.analyzer
                temp_source = self.current_data_source

                if choice == '1':
                    self._load_from_strava_api()
                elif choice == '2':
                    self._load_from_csv()
                elif choice == '3':
                    self._load_from_excel()
                elif choice == '4':
                    self._load_from_cache()

                if self.analyzer and not self.analyzer.df.empty:
                    dfs.append(self.analyzer.df.copy())
                    source_names.append(self.current_data_source)
                    print(f"✓ Added {len(self.analyzer.df)} activities from {self.current_data_source}")

                # Restore
                self.analyzer = temp_analyzer
                self.current_data_source = temp_source
            else:
                print("✗ Invalid option")

        if len(dfs) < 2:
            print("\n✗ Need at least 2 sources to merge")
            return

        # Merge all dataframes
        merged_df = merge_activities(dfs, deduplicate=True)

        if not merged_df.empty:
            source_desc = " + ".join(source_names)
            self._initialize_analyzer(merged_df, f"Merged: {source_desc}")

            # Offer to cache
            cache_it = input("\nSave merged data to cache? (y/n): ").strip().lower()
            if cache_it == 'y':
                cache_source = CachedDataSource(self.unit_system)
                cache_name = input("Cache name [merged_data]: ").strip() or 'merged_data'
                cache_source.save_cache(merged_df, cache_name=cache_name, source=source_desc)

    def _initialize_analyzer(self, df, source_name: str):
        """Initialize analyzer with loaded data."""
        self.analyzer = MarathonTrainingAnalyzer(df, self.unit_system)
        self.visualizer = TrainingVisualizer(self.analyzer)
        self.reporter = TrainingReportGenerator(self.analyzer)
        self.data_loaded = True
        self.current_data_source = source_name
        print(f"\n✓ Successfully loaded {len(df)} activities")

    def _view_summary(self):
        """View training summary."""
        if not self._check_data():
            return

        report = self.reporter.generate_training_report()
        self.reporter.print_report(report, sections_to_print=['summary'])

    def _analyze_progression(self):
        """Analyze training progression."""
        if not self._check_data():
            return

        agg_stats = self.analyzer.calculate_aggregate_stats()
        print(f"\n{self.analyzer.agg_label.upper()} PROGRESSION")
        print("="*70)
        print(agg_stats.to_string())

        show_chart = input("\nShow progression chart? (y/n): ").strip().lower()
        if show_chart == 'y':
            import matplotlib.pyplot as plt
            fig = self.visualizer.plot_progression()
            if fig:
                plt.show()

    def _analyze_long_runs(self):
        """Analyze long runs."""
        if not self._check_data():
            return

        default_dist = self.config.get_long_run_threshold()
        min_dist_str = input(f"\nMinimum distance for long runs [default: {default_dist} {self.unit_system}]: ").strip()
        min_dist = float(min_dist_str) if min_dist_str and min_dist_str.replace('.', '', 1).isdigit() else default_dist

        long_runs = self.analyzer.analyze_long_runs(min_dist)

        if not long_runs.empty:
            print(f"\nLONG RUNS (>= {min_dist} {self.unit_system})")
            print("="*70)
            print(long_runs[['start_date', 'name', 'distance_unit', 'pace_per_unit', 'days_since_last']].to_string())
        else:
            print(f"\n✗ No runs found >= {min_dist} {self.unit_system}")

    def _view_predictions(self):
        """View marathon time predictions."""
        if not self._check_data():
            return

        print("\nNote: Predictions are based on your best race efforts in the last 6 months.")
        report = self.reporter.generate_training_report()
        self.reporter.print_report(report, sections_to_print=['predictions'])

    def _analyze_training_load(self):
        """Analyze training load (ATL/CTL/TSB)."""
        if not self._check_data():
            return

        load_data = self.analyzer.calculate_training_load()

        if load_data.empty:
            print("\n✗ No training load data available")
            return

        # Show summary
        report = self.reporter.generate_training_report()
        self.reporter.print_report(report, sections_to_print=['training_load'])

        # Offer to show chart
        show_chart = input("\nShow training load chart? (y/n): ").strip().lower()
        if show_chart == 'y':
            import webbrowser
            fig = self.visualizer.plot_training_load()
            filename = f"training_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(filename)
            print(f"\n✓ Chart saved to {filename}")
            webbrowser.open(f"file://{os.path.abspath(filename)}")

    def _analyze_heart_rate_zones(self):
        """Analyze heart rate zones."""
        if not self._check_data():
            return

        # Get max HR from config or user
        max_hr = self.config.get('max_heart_rate')
        if not max_hr:
            max_hr_input = input("\nEnter your maximum heart rate (or press Enter to estimate from data): ").strip()
            if max_hr_input:
                try:
                    max_hr = int(max_hr_input)
                except ValueError:
                    max_hr = None

        zone_analysis = self.analyzer.analyze_heart_rate_zones(max_hr=max_hr)

        if not zone_analysis:
            print("\n✗ No heart rate data available in your activities")
            return

        self.reporter.print_hr_zone_report(zone_analysis)

        # Offer to show chart
        show_chart = input("\nShow heart rate zone chart? (y/n): ").strip().lower()
        if show_chart == 'y':
            import webbrowser
            fig = self.visualizer.plot_heart_rate_zones(zone_analysis)
            filename = f"hr_zones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(filename)
            print(f"\n✓ Chart saved to {filename}")
            webbrowser.open(f"file://{os.path.abspath(filename)}")

    def _check_injury_risks(self):
        """Check for injury risk indicators."""
        if not self._check_data():
            return

        risks = self.analyzer.detect_injury_risks()

        print("\n" + "="*70)
        print("INJURY RISK ANALYSIS")
        print("="*70)

        if not risks.get('warnings'):
            print("\n✓ No significant injury risks detected!")
            print("Your training progression looks safe.")
        else:
            print("\n⚠ WARNINGS DETECTED:\n")
            for warning in risks['warnings']:
                print(f"  {warning}")

            if risks.get('rapid_increase'):
                print("\n  Rapid mileage increases can lead to injury.")
                print("  Consider the 10% rule: don't increase weekly mileage by more than 10%.")

            if risks.get('insufficient_recovery'):
                print("\n  Consecutive running days without rest can increase injury risk.")
                print("  Consider adding rest days or easy recovery runs.")

        print("="*70)

    def _compare_periods(self):
        """Compare training between two periods."""
        if not self._check_data():
            return

        print("\nCOMPARE TRAINING PERIODS")
        print("-"*70)

        # Get period 1
        print("\nPeriod 1:")
        p1_start = input("  Start date (YYYY-MM-DD): ").strip()
        p1_end = input("  End date (YYYY-MM-DD): ").strip()

        # Get period 2
        print("\nPeriod 2:")
        p2_start = input("  Start date (YYYY-MM-DD): ").strip()
        p2_end = input("  End date (YYYY-MM-DD): ").strip()

        try:
            p1_start_dt = datetime.strptime(p1_start, '%Y-%m-%d')
            p1_end_dt = datetime.strptime(p1_end, '%Y-%m-%d')
            p2_start_dt = datetime.strptime(p2_start, '%Y-%m-%d')
            p2_end_dt = datetime.strptime(p2_end, '%Y-%m-%d')

            comparison = self.analyzer.compare_periods(p1_start_dt, p1_end_dt, p2_start_dt, p2_end_dt)
            self.reporter.print_comparison_report(comparison)

            # Offer chart
            show_chart = input("\nShow comparison chart? (y/n): ").strip().lower()
            if show_chart == 'y':
                import webbrowser
                fig = self.visualizer.create_comparison_chart(comparison)
                filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                fig.write_html(filename)
                print(f"\n✓ Chart saved to {filename}")
                webbrowser.open(f"file://{os.path.abspath(filename)}")

        except ValueError:
            print("\n✗ Invalid date format. Use YYYY-MM-DD")

    def _generate_full_report(self):
        """Generate and display full report."""
        if not self._check_data():
            return

        report = self.reporter.generate_training_report()
        self.reporter.print_report(report)

        # Offer to save
        save_report = input("\nSave report to file? (y/n): ").strip().lower()
        if save_report == 'y':
            print("\n1. Save as JSON")
            print("2. Save as Markdown")
            print("3. Save both")
            format_choice = input("Select format (1-3): ").strip()

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if format_choice in ['1', '3']:
                filename = f"marathon_report_{timestamp}.json"
                self.reporter.export_to_json(report, filename)

            if format_choice in ['2', '3']:
                filename = f"marathon_report_{timestamp}.md"
                self.reporter.export_to_markdown(report, filename)

    def _create_visualizations(self):
        """Create interactive visualizations."""
        if not self._check_data():
            return

        import webbrowser

        print("\nCREATE VISUALIZATIONS")
        print("-"*70)
        print("1. Training Dashboard")
        print("2. Calendar Heatmap")
        print("3. Training Load Chart")
        print("4. Heart Rate Zones")
        print("5. Elevation Analysis")
        print("6. Pace vs Distance")
        print("7. Create all")
        print("8. Back")

        choice = input("\nSelect visualization (1-8): ").strip()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if choice in ['1', '7']:
            fig = self.visualizer.create_training_dashboard()
            filename = f"dashboard_{timestamp}.html"
            fig.write_html(filename)
            print(f"✓ Dashboard saved to {filename}")
            if choice == '1':
                webbrowser.open(f"file://{os.path.abspath(filename)}")

        if choice in ['2', '7']:
            fig = self.visualizer.create_calendar_heatmap()
            filename = f"calendar_{timestamp}.html"
            fig.write_html(filename)
            print(f"✓ Calendar heatmap saved to {filename}")
            if choice == '2':
                webbrowser.open(f"file://{os.path.abspath(filename)}")

        if choice in ['3', '7']:
            fig = self.visualizer.plot_training_load()
            filename = f"training_load_{timestamp}.html"
            fig.write_html(filename)
            print(f"✓ Training load chart saved to {filename}")
            if choice == '3':
                webbrowser.open(f"file://{os.path.abspath(filename)}")

        if choice in ['4', '7']:
            zone_analysis = self.analyzer.analyze_heart_rate_zones()
            if zone_analysis:
                fig = self.visualizer.plot_heart_rate_zones(zone_analysis)
                filename = f"hr_zones_{timestamp}.html"
                fig.write_html(filename)
                print(f"✓ HR zones chart saved to {filename}")
                if choice == '4':
                    webbrowser.open(f"file://{os.path.abspath(filename)}")
            else:
                print("✗ No heart rate data available")

        if choice in ['5', '7']:
            fig = self.visualizer.plot_elevation_analysis()
            if fig.data:
                filename = f"elevation_{timestamp}.html"
                fig.write_html(filename)
                print(f"✓ Elevation analysis saved to {filename}")
                if choice == '5':
                    webbrowser.open(f"file://{os.path.abspath(filename)}")
            else:
                print("✗ No elevation data available")

        if choice in ['6', '7']:
            fig = self.visualizer.plot_pace_vs_distance()
            filename = f"pace_distance_{timestamp}.html"
            fig.write_html(filename)
            print(f"✓ Pace vs distance chart saved to {filename}")
            if choice == '6':
                webbrowser.open(f"file://{os.path.abspath(filename)}")

    def _manage_goals(self):
        """Manage training goals."""
        while True:
            print("\n" + "-"*70)
            print("MANAGE GOALS")
            print("-"*70)

            goals = self.config.get_goals()
            if goals:
                for i, goal in enumerate(goals, 1):
                    print(f"{i}. {goal.get('race_name', 'Unnamed')} - {goal.get('race_date', 'TBD')}")
                    if goal.get('goal_time_minutes'):
                        t = goal['goal_time_minutes']
                        h, m = int(t // 60), int(t % 60)
                        print(f"   Goal time: {h}:{m:02d}")
            else:
                print("No goals set")

            print("\nOptions:")
            print("1. Add new goal")
            print("2. Remove goal")
            print("3. View upcoming goal")
            print("4. Back")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                self._add_goal()
            elif choice == '2':
                self._remove_goal()
            elif choice == '3':
                self._view_upcoming_goal()
            elif choice == '4':
                break

    def _add_goal(self):
        """Add a training goal."""
        print("\nADD NEW GOAL")
        print("-"*70)

        race_name = input("Race name: ").strip()
        race_date = input("Race date (YYYY-MM-DD): ").strip()

        try:
            # Validate date
            datetime.strptime(race_date, '%Y-%m-%d')
        except ValueError:
            print("✗ Invalid date format")
            return

        race_distance = input(f"Race distance ({self.unit_system}): ").strip()
        try:
            race_distance = float(race_distance)
        except ValueError:
            print("✗ Invalid distance")
            return

        goal_time = input("Goal time in minutes (optional, press Enter to skip): ").strip()
        goal_time_minutes = None
        if goal_time:
            try:
                goal_time_minutes = float(goal_time)
            except ValueError:
                print("Invalid time, skipping")

        notes = input("Notes (optional): ").strip()

        goal = {
            'race_name': race_name,
            'race_date': race_date,
            'race_distance': race_distance,
            'goal_time_minutes': goal_time_minutes,
            'notes': notes
        }

        self.config.add_goal(goal)
        print("\n✓ Goal added!")

    def _remove_goal(self):
        """Remove a goal."""
        goals = self.config.get_goals()
        if not goals:
            print("\nNo goals to remove")
            return

        try:
            idx = int(input("\nEnter goal number to remove: ")) - 1
            if self.config.remove_goal(idx):
                print("✓ Goal removed")
            else:
                print("✗ Invalid goal number")
        except ValueError:
            print("✗ Invalid input")

    def _view_upcoming_goal(self):
        """View next upcoming goal."""
        goal = self.config.get_upcoming_goal()

        if not goal:
            print("\nNo upcoming goals")
            return

        print("\n" + "="*70)
        print("NEXT UPCOMING RACE")
        print("="*70)
        print(f"Race: {goal.get('race_name', 'Unnamed')}")
        print(f"Date: {goal.get('race_date', 'TBD')}")
        print(f"Distance: {goal.get('race_distance', '?')} {self.unit_system}")

        if goal.get('goal_time_minutes'):
            t = goal['goal_time_minutes']
            h, m = int(t // 60), int(t % 60)
            print(f"Goal Time: {h}:{m:02d}")

        if goal.get('notes'):
            print(f"Notes: {goal['notes']}")

        # Calculate days until race
        try:
            race_date = datetime.strptime(goal['race_date'], '%Y-%m-%d').date()
            days_until = (race_date - datetime.now().date()).days
            print(f"\nDays until race: {days_until}")
        except Exception:
            pass

        print("="*70)

    def _export_data(self):
        """Export activity data."""
        if not self._check_data():
            return

        print("\nEXPORT DATA")
        print("-"*70)
        print("1. Export to CSV")
        print("2. Export to Excel")
        print("3. Export both")

        choice = input("\nSelect format (1-3): ").strip()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if choice in ['1', '3']:
            filename = f"training_data_{timestamp}.csv"
            self.analyzer.df.to_csv(filename, index=False)
            print(f"✓ Data exported to {filename}")

        if choice in ['2', '3']:
            filename = f"training_data_{timestamp}.xlsx"
            self.analyzer.df.to_excel(filename, index=False)
            print(f"✓ Data exported to {filename}")

    def _settings_menu(self):
        """Settings menu."""
        while True:
            print("\n" + "-"*70)
            print("SETTINGS")
            print("-"*70)
            print("1. View current settings")
            print("2. Update settings")
            print("3. Run setup wizard")
            print("4. Back")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                self.config.display_settings()
            elif choice == '2':
                self.config.update_interactive()
                # Reload unit system if changed
                self.unit_system = self.config.get_unit_system()
            elif choice == '3':
                self.config.setup_wizard()
                self.unit_system = self.config.get_unit_system()
            elif choice == '4':
                break

    def _parse_time_period(self, period_str: str) -> Optional[timedelta]:
        """Parse time period string (e.g., '12w', '6m', '1y')."""
        match = re.match(r"(\d+)([wmy])", period_str.lower())
        if not match:
            return None

        value, unit = int(match.group(1)), match.group(2)

        if unit == 'w':
            return timedelta(weeks=value)
        elif unit == 'm':
            return timedelta(days=value * 30)
        elif unit == 'y':
            return timedelta(days=value * 365)

        return None

    def _check_data(self) -> bool:
        """Check if data is loaded."""
        if not self.data_loaded or self.analyzer is None or self.analyzer.df.empty:
            print("\n✗ No data loaded. Please load data first (Option 1).")
            return False
        return True


def main():
    """Main entry point."""
    app = MarathonTrainingApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n✓ Application terminated by user.")
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
