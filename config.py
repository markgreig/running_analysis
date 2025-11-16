"""
Configuration management for Marathon Training Analyzer.
Handles user preferences, settings persistence, and goals.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class Config:
    """Manages application configuration and user preferences."""

    CONFIG_DIR = Path.home() / '.marathon_analyzer'
    CONFIG_FILE = CONFIG_DIR / 'config.json'

    DEFAULT_CONFIG = {
        'unit_system': 'km',
        'default_fetch_period': '12w',
        'long_run_threshold_km': 16.0,
        'long_run_threshold_miles': 10.0,
        'max_heart_rate': None,
        'resting_heart_rate': None,
        'cache_enabled': True,
        'auto_save_reports': False,
        'theme': 'default',
        'goals': []
    }

    def __init__(self):
        """Initialize configuration manager."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """
        Load configuration from file.

        Returns:
            Configuration dictionary
        """
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                # Merge with defaults to handle new config keys
                config = self.DEFAULT_CONFIG.copy()
                config.update(loaded)
                return config
            except Exception as e:
                print(f"Warning: Failed to load config: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            return self.DEFAULT_CONFIG.copy()

    def save(self):
        """Save configuration to file."""
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def get(self, key: str, default=None):
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key doesn't exist

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value):
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value

    def get_unit_system(self) -> str:
        """Get preferred unit system."""
        return self.config.get('unit_system', 'km')

    def set_unit_system(self, unit_system: str):
        """Set preferred unit system."""
        if unit_system in ['km', 'miles']:
            self.config['unit_system'] = unit_system
            self.save()

    def get_long_run_threshold(self, unit_system: Optional[str] = None) -> float:
        """
        Get long run threshold for unit system.

        Args:
            unit_system: 'km' or 'miles' (default: configured unit system)

        Returns:
            Long run threshold distance
        """
        if unit_system is None:
            unit_system = self.get_unit_system()

        if unit_system == 'km':
            return self.config.get('long_run_threshold_km', 16.0)
        else:
            return self.config.get('long_run_threshold_miles', 10.0)

    def set_long_run_threshold(self, threshold: float, unit_system: Optional[str] = None):
        """Set long run threshold."""
        if unit_system is None:
            unit_system = self.get_unit_system()

        if unit_system == 'km':
            self.config['long_run_threshold_km'] = threshold
        else:
            self.config['long_run_threshold_miles'] = threshold
        self.save()

    def add_goal(self, goal: Dict):
        """
        Add a training goal.

        Args:
            goal: Dictionary with goal details
                - race_date: str (YYYY-MM-DD)
                - race_name: str
                - race_distance: float
                - goal_time_minutes: float (optional)
                - notes: str (optional)
        """
        if 'goals' not in self.config:
            self.config['goals'] = []

        goal['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.config['goals'].append(goal)
        self.save()

    def get_goals(self) -> list:
        """Get all training goals."""
        return self.config.get('goals', [])

    def remove_goal(self, index: int) -> bool:
        """
        Remove a goal by index.

        Args:
            index: Goal index

        Returns:
            True if successful
        """
        if 'goals' in self.config and 0 <= index < len(self.config['goals']):
            self.config['goals'].pop(index)
            self.save()
            return True
        return False

    def get_upcoming_goal(self) -> Optional[Dict]:
        """
        Get the next upcoming race goal.

        Returns:
            Goal dictionary or None
        """
        goals = self.get_goals()
        if not goals:
            return None

        # Filter future goals
        today = datetime.now().date()
        future_goals = []

        for goal in goals:
            try:
                race_date = datetime.strptime(goal['race_date'], '%Y-%m-%d').date()
                if race_date >= today:
                    goal['_race_date_obj'] = race_date
                    future_goals.append(goal)
            except Exception:
                continue

        if not future_goals:
            return None

        # Sort by date and return earliest
        future_goals.sort(key=lambda g: g['_race_date_obj'])
        return future_goals[0]

    def setup_wizard(self):
        """Run interactive setup wizard for first-time configuration."""
        print("\n" + "="*60)
        print("MARATHON ANALYZER - FIRST TIME SETUP")
        print("="*60)

        # Unit system
        while True:
            unit = input("\nPreferred unit system (km/miles) [km]: ").strip().lower()
            if not unit:
                unit = 'km'
            if unit in ['km', 'miles']:
                self.set_unit_system(unit)
                break
            print("Invalid input. Please enter 'km' or 'miles'.")

        # Long run threshold
        default_threshold = 16.0 if unit == 'km' else 10.0
        threshold_input = input(f"\nLong run minimum distance [{default_threshold} {unit}]: ").strip()
        if threshold_input:
            try:
                threshold = float(threshold_input)
                self.set_long_run_threshold(threshold, unit)
            except ValueError:
                print(f"Invalid input. Using default: {default_threshold} {unit}")

        # Heart rate
        max_hr_input = input("\nMaximum heart rate (optional, press Enter to skip): ").strip()
        if max_hr_input:
            try:
                max_hr = int(max_hr_input)
                self.config['max_heart_rate'] = max_hr
            except ValueError:
                print("Invalid input. Skipping.")

        resting_hr_input = input("Resting heart rate (optional, press Enter to skip): ").strip()
        if resting_hr_input:
            try:
                resting_hr = int(resting_hr_input)
                self.config['resting_heart_rate'] = resting_hr
            except ValueError:
                print("Invalid input. Skipping.")

        # Save configuration
        self.save()
        print("\n✓ Configuration saved!")

    def display_settings(self):
        """Display current settings."""
        print("\n" + "="*60)
        print("CURRENT SETTINGS")
        print("="*60)
        print(f"Unit System: {self.get_unit_system()}")
        print(f"Long Run Threshold: {self.get_long_run_threshold()} {self.get_unit_system()}")
        print(f"Max Heart Rate: {self.config.get('max_heart_rate', 'Not set')}")
        print(f"Resting Heart Rate: {self.config.get('resting_heart_rate', 'Not set')}")
        print(f"Default Fetch Period: {self.config.get('default_fetch_period', '12w')}")
        print(f"Cache Enabled: {self.config.get('cache_enabled', True)}")
        print(f"Auto-save Reports: {self.config.get('auto_save_reports', False)}")

        goals = self.get_goals()
        if goals:
            print(f"\nActive Goals: {len(goals)}")
            for i, goal in enumerate(goals):
                print(f"  {i+1}. {goal.get('race_name', 'Unnamed')} on {goal.get('race_date', 'TBD')}")
        else:
            print("\nNo active goals")

        print("="*60)

    def update_interactive(self):
        """Interactive configuration update."""
        print("\n" + "="*60)
        print("UPDATE SETTINGS")
        print("="*60)
        print("\n1. Unit System")
        print("2. Long Run Threshold")
        print("3. Max Heart Rate")
        print("4. Resting Heart Rate")
        print("5. Default Fetch Period")
        print("6. Toggle Cache")
        print("7. Toggle Auto-save Reports")
        print("8. Back")

        choice = input("\nSelect setting to update (1-8): ").strip()

        if choice == '1':
            unit = input("Enter unit system (km/miles): ").strip().lower()
            if unit in ['km', 'miles']:
                self.set_unit_system(unit)
                print("✓ Unit system updated")
        elif choice == '2':
            try:
                threshold = float(input(f"Enter long run threshold ({self.get_unit_system()}): "))
                self.set_long_run_threshold(threshold)
                print("✓ Long run threshold updated")
            except ValueError:
                print("✗ Invalid input")
        elif choice == '3':
            try:
                max_hr = int(input("Enter max heart rate: "))
                self.config['max_heart_rate'] = max_hr
                self.save()
                print("✓ Max heart rate updated")
            except ValueError:
                print("✗ Invalid input")
        elif choice == '4':
            try:
                resting_hr = int(input("Enter resting heart rate: "))
                self.config['resting_heart_rate'] = resting_hr
                self.save()
                print("✓ Resting heart rate updated")
            except ValueError:
                print("✗ Invalid input")
        elif choice == '5':
            period = input("Enter default fetch period (e.g., 12w, 6m, 1y): ").strip()
            self.config['default_fetch_period'] = period
            self.save()
            print("✓ Default fetch period updated")
        elif choice == '6':
            self.config['cache_enabled'] = not self.config.get('cache_enabled', True)
            self.save()
            print(f"✓ Cache {'enabled' if self.config['cache_enabled'] else 'disabled'}")
        elif choice == '7':
            self.config['auto_save_reports'] = not self.config.get('auto_save_reports', False)
            self.save()
            print(f"✓ Auto-save reports {'enabled' if self.config['auto_save_reports'] else 'disabled'}")
