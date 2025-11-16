"""
Report generation for marathon training analysis.
Creates formatted console reports and exportable files.
"""

from typing import Dict, List, Optional
from datetime import datetime

from analyzers import MarathonTrainingAnalyzer


class TrainingReportGenerator:
    """Generates formatted reports from training analysis."""

    def __init__(self, analyzer: MarathonTrainingAnalyzer):
        """
        Initialize report generator.

        Args:
            analyzer: MarathonTrainingAnalyzer instance with data
        """
        self.analyzer = analyzer
        self.unit = self.analyzer.unit_system

    def generate_training_report(self) -> Dict:
        """
        Generate comprehensive training report.

        Returns:
            Dictionary with all report sections
        """
        report = {
            'summary': {},
            'aggregate_analysis': {},
            'long_run_analysis': {},
            'predictions': {},
            'training_load': {},
            'injury_risks': {},
            'hr_zones': {}
        }

        # Summary statistics
        report['summary'] = self.analyzer.get_training_summary()

        # Aggregate analysis
        agg_stats = self.analyzer.calculate_aggregate_stats()
        if not agg_stats.empty:
            report['aggregate_analysis'] = {
                f'peak_{self.analyzer.agg_label}_distance_{self.unit}': round(agg_stats['total_distance'].max(), 1),
                'consistency_score_pct': round(100 * (1 - agg_stats['total_distance'].std() / agg_stats['total_distance'].mean()), 1) if agg_stats['total_distance'].mean() > 0 else 0,
                f'avg_runs_per_{self.analyzer.agg_label.lower()[:-2]}': round(agg_stats['run_count'].mean(), 1)
            }

        # Long run analysis
        long_runs = self.analyzer.analyze_long_runs()
        if not long_runs.empty:
            report['long_run_analysis'] = {
                'total_long_runs': len(long_runs),
                f'longest_run_{self.unit}': round(long_runs['distance_unit'].max(), 1),
                f'avg_long_run_distance_{self.unit}': round(long_runs['distance_unit'].mean(), 1),
                'avg_days_between_long_runs': round(long_runs['days_since_last'].mean(), 1) if 'days_since_last' in long_runs.columns else None
            }

        # Predictions
        report['predictions'] = self.analyzer.predict_marathon_time()

        # Training load
        load_data = self.analyzer.calculate_training_load()
        if not load_data.empty:
            latest = load_data.iloc[-1]
            report['training_load'] = {
                'current_ATL_fatigue': round(latest['ATL'], 1),
                'current_CTL_fitness': round(latest['CTL'], 1),
                'current_TSB_freshness': round(latest['TSB'], 1),
                'status': self._interpret_tsb(latest['TSB'])
            }

        # Injury risks
        report['injury_risks'] = self.analyzer.detect_injury_risks()

        return report

    def _interpret_tsb(self, tsb: float) -> str:
        """Interpret Training Stress Balance value."""
        if tsb < -30:
            return "Very High Fatigue - Risk of overtraining"
        elif tsb < -10:
            return "High Fatigue - Productive training load"
        elif tsb < 5:
            return "Optimal - Good balance"
        elif tsb < 15:
            return "Fresh - Ready to race"
        else:
            return "Very Fresh - Possible detraining"

    def print_report(self, report: Dict, sections_to_print: Optional[List[str]] = None):
        """
        Print formatted report to console.

        Args:
            report: Report dictionary from generate_training_report()
            sections_to_print: List of section names to print (default: all)
        """
        if not sections_to_print:
            sections_to_print = [
                'summary', 'aggregate_analysis', 'long_run_analysis',
                'training_load', 'injury_risks', 'predictions'
            ]

        print("\n" + "="*70)
        print(" " * 20 + "TRAINING REPORT")
        print("="*70)

        def format_key(key):
            """Format dictionary key for display."""
            return key.replace(f'_{self.unit}', '').replace('_', ' ').title()

        for section in sections_to_print:
            if section not in report or not report[section]:
                continue

            self._print_section(section, report[section], format_key)

        print("\n" + "="*70)

    def _print_section(self, section_name: str, section_data: Dict, format_key):
        """Print a single section of the report."""
        title = section_name.replace('_', ' ').upper()
        print(f"\n{title}")
        print("-" * 70)

        if section_name == 'predictions':
            self._print_predictions(section_data)
        elif section_name == 'injury_risks':
            self._print_injury_risks(section_data)
        elif section_name == 'training_load':
            self._print_training_load(section_data)
        else:
            for key, value in section_data.items():
                if value is not None:
                    print(f"  {format_key(key)}: {value}")

    def _print_predictions(self, predictions: Dict):
        """Print marathon predictions section."""
        if not predictions:
            print("  Not enough race data for predictions.")
            return

        if 'Average' in predictions:
            avg_pred = predictions['Average']
            t = avg_pred['predicted_time_minutes']
            h, m, s = int(t // 60), int(t % 60), int((t * 60) % 60)
            print(f"\n  PREDICTED MARATHON TIME: {h}:{m:02d}:{s:02d}")

            p = avg_pred['predicted_pace']
            p_min, p_sec = int(p), int((p * 60) % 60)
            print(f"  Target Pace: {p_min}:{p_sec:02d} per {self.unit}")

            if 'confidence_range_minutes' in avg_pred:
                range_min = avg_pred['confidence_range_minutes']
                print(f"  Prediction Range: ± {int(range_min)} minutes")

            print("\n  Individual Model Predictions:")
            for model_name, pred in predictions.items():
                if model_name != 'Average':
                    t = pred['predicted_time_minutes']
                    h, m, s = int(t // 60), int(t % 60), int((t * 60) % 60)
                    source = pred.get('source_race', 'N/A')
                    model = pred.get('model', 'N/A')
                    print(f"    • {model_name}: {h}:{m:02d}:{s:02d} (via {source})")
        else:
            print("  Not enough recent race data for prediction.")

    def _print_injury_risks(self, risks: Dict):
        """Print injury risk warnings."""
        if not risks.get('warnings'):
            print("  ✓ No significant injury risks detected")
            return

        for warning in risks['warnings']:
            print(f"  {warning}")

        if risks.get('rapid_increase'):
            print("\n  Rapid Mileage Increases:")
            for incident in risks['rapid_increase'][:5]:  # Show top 5
                print(f"    • {incident['period']}: +{incident['increase_pct']:.1f}% increase")

    def _print_training_load(self, load_data: Dict):
        """Print training load metrics."""
        if not load_data:
            print("  No training load data available")
            return

        print(f"  Current Fitness (CTL): {load_data.get('current_CTL_fitness', 'N/A')}")
        print(f"  Current Fatigue (ATL): {load_data.get('current_ATL_fatigue', 'N/A')}")
        print(f"  Freshness (TSB): {load_data.get('current_TSB_freshness', 'N/A')}")
        print(f"  Status: {load_data.get('status', 'N/A')}")

    def print_comparison_report(self, comparison: Dict):
        """
        Print comparison report between two periods.

        Args:
            comparison: Comparison dictionary from analyzer.compare_periods()
        """
        if not comparison:
            print("No comparison data available")
            return

        print("\n" + "="*70)
        print(" " * 20 + "TRAINING COMPARISON")
        print("="*70)

        p1 = comparison['period1']
        p2 = comparison['period2']

        print(f"\nPeriod 1: {p1['start']} to {p1['end']}")
        self._print_period_stats(p1['stats'])

        print(f"\nPeriod 2: {p2['start']} to {p2['end']}")
        self._print_period_stats(p2['stats'])

        print("\nChanges:")
        print("-" * 70)
        for metric, change in comparison['changes'].items():
            metric_name = metric.replace('_', ' ').title()
            abs_change = change['absolute']
            pct_change = change['percent']
            direction = "↑" if pct_change > 0 else "↓" if pct_change < 0 else "="
            print(f"  {metric_name}: {direction} {abs(abs_change):.1f} ({pct_change:+.1f}%)")

        print("="*70)

    def _print_period_stats(self, stats: Dict):
        """Print statistics for a single period."""
        if not stats:
            print("  No data")
            return

        print(f"  Total Runs: {stats.get('run_count', 'N/A')}")
        print(f"  Total Distance: {stats.get('total_distance', 'N/A'):.1f} {self.unit}")
        print(f"  Average Distance: {stats.get('avg_distance', 'N/A'):.1f} {self.unit}")
        print(f"  Total Time: {stats.get('total_time_hours', 'N/A'):.1f} hours")
        print(f"  Average Pace: {stats.get('avg_pace', 'N/A'):.2f} min/{self.unit}")
        print(f"  Longest Run: {stats.get('longest_run', 'N/A'):.1f} {self.unit}")

    def print_hr_zone_report(self, zone_analysis: Dict):
        """
        Print heart rate zone analysis report.

        Args:
            zone_analysis: Zone analysis from analyzer.analyze_heart_rate_zones()
        """
        if not zone_analysis:
            print("No heart rate data available")
            return

        print("\n" + "="*70)
        print(" " * 18 + "HEART RATE ZONE ANALYSIS")
        print("="*70)

        print(f"\nMax Heart Rate Used: {zone_analysis.get('max_hr_used', 'N/A')} bpm")
        print(f"Total Runs with HR Data: {zone_analysis.get('total_runs_with_hr', 'N/A')}")

        print("\nZone Distribution:")
        print("-" * 70)

        for zone_name, data in zone_analysis.items():
            if zone_name not in ['max_hr_used', 'total_runs_with_hr']:
                print(f"\n{zone_name} - {data['hr_range']}")
                print(f"  Runs: {data['run_count']}")
                print(f"  Time: {data['total_time_minutes']:.0f} minutes ({data['pct_of_runs']:.1f}% of total runs)")
                print(f"  Distance: {data['total_distance']:.1f} {self.unit}")

        print("\n" + "="*70)

        # Add interpretation
        print("\nInterpretation:")
        print("-" * 70)
        print("  Zone 1-2: Base building, recovery runs")
        print("  Zone 3: Aerobic endurance, tempo runs")
        print("  Zone 4: Lactate threshold training")
        print("  Zone 5: VO2 max intervals")
        print("\n  Recommended: 80% easy (Z1-Z2), 20% hard (Z3-Z5)")
        print("="*70)

    def export_to_json(self, report: Dict, filename: str):
        """
        Export report to JSON file.

        Args:
            report: Report dictionary
            filename: Output filename
        """
        import json

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✓ Report exported to {filename}")
        except Exception as e:
            print(f"✗ Failed to export report: {e}")

    def export_to_markdown(self, report: Dict, filename: str):
        """
        Export report to Markdown file.

        Args:
            report: Report dictionary
            filename: Output filename
        """
        try:
            with open(filename, 'w') as f:
                f.write("# Marathon Training Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Summary
                if report.get('summary'):
                    f.write("## Training Summary\n\n")
                    for key, value in report['summary'].items():
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                    f.write("\n")

                # Predictions
                if report.get('predictions') and report['predictions'].get('Average'):
                    f.write("## Marathon Time Prediction\n\n")
                    avg = report['predictions']['Average']
                    t = avg['predicted_time_minutes']
                    h, m = int(t // 60), int(t % 60)
                    f.write(f"**Predicted Time**: {h}:{m:02d}\n\n")

                # Injury Risks
                if report.get('injury_risks') and report['injury_risks'].get('warnings'):
                    f.write("## Injury Risk Warnings\n\n")
                    for warning in report['injury_risks']['warnings']:
                        f.write(f"- {warning}\n")
                    f.write("\n")

            print(f"✓ Report exported to {filename}")
        except Exception as e:
            print(f"✗ Failed to export report: {e}")
