"""
Advanced visualization components for marathon training analysis.
Creates interactive and static charts for comprehensive data analysis.
"""

from typing import Optional, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from analyzers import MarathonTrainingAnalyzer


class TrainingVisualizer:
    """Creates comprehensive visualizations for training analysis."""

    def __init__(self, analyzer: MarathonTrainingAnalyzer):
        """
        Initialize visualizer with analyzer instance.

        Args:
            analyzer: MarathonTrainingAnalyzer instance with data
        """
        self.analyzer = analyzer
        self.unit = self.analyzer.unit_system

    def create_training_dashboard(self) -> go.Figure:
        """
        Create interactive Plotly dashboard with key metrics.

        Returns:
            Plotly Figure with multiple subplots
        """
        agg_stats = self.analyzer.calculate_aggregate_stats()
        agg_label = self.analyzer.agg_label

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{agg_label} Distance ({self.unit})',
                f'Long Run Progression ({self.unit})',
                'Pace Distribution',
                f'{agg_label} Run Count'
            ),
        )

        # Plot 1: Weekly/Monthly distance
        if not agg_stats.empty:
            fig.add_trace(
                go.Scatter(
                    x=agg_stats.index.astype(str),
                    y=agg_stats['total_distance'],
                    mode='lines+markers',
                    name='Distance',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )

        # Plot 2: Long run progression
        long_runs = self.analyzer.analyze_long_runs()
        if not long_runs.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_runs['start_date'],
                    y=long_runs['distance_unit'],
                    mode='lines+markers',
                    name='Long Runs',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )

        # Plot 3: Pace distribution
        if not self.analyzer.df.empty:
            fig.add_trace(
                go.Histogram(
                    x=self.analyzer.df['pace_per_unit'],
                    name='Pace',
                    marker=dict(color='#2ca02c'),
                    nbinsx=30
                ),
                row=2, col=1
            )

        # Plot 4: Run count
        if not agg_stats.empty:
            fig.add_trace(
                go.Bar(
                    x=agg_stats.index.astype(str),
                    y=agg_stats['run_count'],
                    name='Runs',
                    marker=dict(color='#d62728')
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            title_text="Training Analysis Dashboard",
            showlegend=False
        )

        fig.update_xaxes(title_text="Period", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text=f"Pace (min/{self.unit})", row=2, col=1)
        fig.update_xaxes(title_text="Period", row=2, col=2)

        fig.update_yaxes(title_text=f"Distance ({self.unit})", row=1, col=1)
        fig.update_yaxes(title_text=f"Distance ({self.unit})", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Runs", row=2, col=2)

        return fig

    def plot_progression(self) -> plt.Figure:
        """
        Create matplotlib figure with training progression analysis.

        Returns:
            Matplotlib Figure
        """
        agg_stats = self.analyzer.calculate_aggregate_stats()
        agg_label = self.analyzer.agg_label

        if agg_stats is None or agg_stats.empty:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{agg_label} Training Progression', fontsize=16)

        # Plot 1: Distance progression
        axes[0, 0].plot(agg_stats.index.astype(str), agg_stats['total_distance'], marker='o', linewidth=2)
        axes[0, 0].set_title(f'{agg_label} Distance Progression')
        axes[0, 0].set_ylabel(f'Distance ({self.unit})')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Run frequency
        axes[0, 1].bar(agg_stats.index.astype(str), agg_stats['run_count'], color='skyblue')
        axes[0, 1].set_title(f'{agg_label} Run Frequency')
        axes[0, 1].set_ylabel('Number of Runs')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Plot 3: Average pace
        if 'avg_pace' in agg_stats.columns:
            axes[1, 0].plot(agg_stats.index.astype(str), agg_stats['avg_pace'], marker='s', color='green', linewidth=2)
            axes[1, 0].set_title(f'Average {agg_label} Pace')
            axes[1, 0].set_ylabel(f'Minutes per {self.unit}')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].invert_yaxis()  # Lower pace is better

        # Plot 4: Distance change
        colors = ['red' if x < 0 else 'green' for x in agg_stats['distance_change'].fillna(0)]
        axes[1, 1].bar(agg_stats.index.astype(str), agg_stats['distance_change'].fillna(0), color=colors)
        axes[1, 1].set_title(f'Period-over-Period Distance Change (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].axhline(y=10, color='red', linestyle='--', linewidth=1, label='10% threshold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def create_calendar_heatmap(self) -> go.Figure:
        """
        Create a calendar heatmap showing running activity (GitHub-style).

        Returns:
            Plotly Figure with calendar heatmap
        """
        if self.analyzer.df.empty:
            return go.Figure()

        # Aggregate by date
        daily = self.analyzer.df.groupby(self.analyzer.df['start_date'].dt.date).agg({
            'distance_unit': 'sum'
        }).reset_index()

        daily.columns = ['date', 'distance']
        daily['date'] = pd.to_datetime(daily['date'])

        # Create full date range
        date_range = pd.date_range(
            start=daily['date'].min(),
            end=daily['date'].max(),
            freq='D'
        )

        # Reindex to include all dates
        daily_full = daily.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily_full.columns = ['date', 'distance']

        # Prepare data for heatmap
        daily_full['week'] = daily_full['date'].dt.isocalendar().week
        daily_full['year'] = daily_full['date'].dt.year
        daily_full['weekday'] = daily_full['date'].dt.dayofweek
        daily_full['month'] = daily_full['date'].dt.month

        # Create pivot table
        pivot = daily_full.pivot_table(
            index='weekday',
            columns='week',
            values='distance',
            aggfunc='sum',
            fill_value=0
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale='Greens',
            hovertemplate=f'Week: %{{x}}<br>Day: %{{y}}<br>Distance: %{{z:.1f}} {self.unit}<extra></extra>',
            colorbar=dict(title=f'Distance ({self.unit})')
        ))

        fig.update_layout(
            title='Training Calendar Heatmap',
            xaxis_title='Week of Year',
            yaxis_title='Day of Week',
            height=300,
        )

        return fig

    def plot_training_load(self) -> go.Figure:
        """
        Create training load chart (ATL, CTL, TSB).

        Returns:
            Plotly Figure with training load metrics
        """
        load_data = self.analyzer.calculate_training_load()

        if load_data.empty:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Load (Fitness & Fatigue)', 'Training Stress Balance (Freshness)'),
            vertical_spacing=0.12
        )

        # Plot ATL and CTL
        fig.add_trace(
            go.Scatter(
                x=load_data['date'],
                y=load_data['ATL'],
                name='ATL (Fatigue)',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=load_data['date'],
                y=load_data['CTL'],
                name='CTL (Fitness)',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Plot TSB
        colors = ['red' if x < -10 else 'green' if x > 5 else 'gray' for x in load_data['TSB']]
        fig.add_trace(
            go.Bar(
                x=load_data['date'],
                y=load_data['TSB'],
                name='TSB',
                marker=dict(color=colors),
                showlegend=False
            ),
            row=2, col=1
        )

        # Add reference lines for TSB
        fig.add_hline(y=0, line=dict(color='black', width=1), row=2, col=1)
        fig.add_hline(y=-10, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
        fig.add_hline(y=5, line=dict(color='green', width=1, dash='dash'), row=2, col=1)

        fig.update_layout(
            height=700,
            title_text="Training Load Analysis",
            showlegend=True
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Training Load", row=1, col=1)
        fig.update_yaxes(title_text="TSB (Freshness)", row=2, col=1)

        return fig

    def plot_heart_rate_zones(self, zone_analysis: Dict) -> go.Figure:
        """
        Create heart rate zone distribution chart.

        Args:
            zone_analysis: Dictionary from analyze_heart_rate_zones()

        Returns:
            Plotly Figure with HR zone distribution
        """
        if not zone_analysis or 'max_hr_used' not in zone_analysis:
            return go.Figure()

        # Extract zone data
        zones = []
        times = []
        distances = []
        run_counts = []

        for zone_name, data in zone_analysis.items():
            if zone_name not in ['max_hr_used', 'total_runs_with_hr']:
                zones.append(zone_name)
                times.append(data['total_time_minutes'])
                distances.append(data['total_distance'])
                run_counts.append(data['run_count'])

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Time in Each Zone', 'Distance in Each Zone'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Time by zone
        fig.add_trace(
            go.Bar(
                x=zones,
                y=times,
                name='Time (minutes)',
                marker=dict(color=['#90EE90', '#FFD700', '#FFA500', '#FF6347', '#DC143C'])
            ),
            row=1, col=1
        )

        # Distance by zone
        fig.add_trace(
            go.Bar(
                x=zones,
                y=distances,
                name=f'Distance ({self.unit})',
                marker=dict(color=['#90EE90', '#FFD700', '#FFA500', '#FF6347', '#DC143C']),
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            title_text=f"Heart Rate Zone Distribution (Max HR: {zone_analysis['max_hr_used']} bpm)",
        )

        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_yaxes(title_text="Minutes", row=1, col=1)
        fig.update_yaxes(title_text=f"Distance ({self.unit})", row=1, col=2)

        return fig

    def plot_elevation_analysis(self) -> go.Figure:
        """
        Create elevation gain analysis charts.

        Returns:
            Plotly Figure with elevation analysis
        """
        if self.analyzer.df.empty or 'elevation_gain' not in self.analyzer.df.columns:
            return go.Figure()

        df_with_elev = self.analyzer.df[self.analyzer.df['elevation_gain'].notna()].copy()

        if df_with_elev.empty:
            return go.Figure()

        # Calculate elevation per unit distance
        df_with_elev['elev_per_unit'] = df_with_elev['elevation_gain'] / df_with_elev['distance_unit']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Elevation Gain Over Time', 'Elevation Gain vs Pace'),
        )

        # Plot 1: Elevation over time
        fig.add_trace(
            go.Scatter(
                x=df_with_elev['start_date'],
                y=df_with_elev['elevation_gain'],
                mode='markers',
                name='Elevation Gain',
                marker=dict(
                    size=8,
                    color=df_with_elev['elevation_gain'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Elevation (m)', x=0.46)
                ),
                text=df_with_elev['name'],
                hovertemplate='%{text}<br>Date: %{x}<br>Elevation: %{y:.0f}m<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Elevation vs Pace
        fig.add_trace(
            go.Scatter(
                x=df_with_elev['elevation_gain'],
                y=df_with_elev['pace_per_unit'],
                mode='markers',
                name='Pace vs Elevation',
                marker=dict(size=8, color='coral'),
                text=df_with_elev['name'],
                hovertemplate='%{text}<br>Elevation: %{x:.0f}m<br>Pace: %{y:.2f} min/%s<extra></extra>' % self.unit,
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            title_text="Elevation Analysis",
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Elevation Gain (m)", row=1, col=2)
        fig.update_yaxes(title_text="Elevation (m)", row=1, col=1)
        fig.update_yaxes(title_text=f"Pace (min/{self.unit})", row=1, col=2)

        return fig

    def plot_pace_vs_distance(self) -> go.Figure:
        """
        Create scatter plot of pace vs distance to identify patterns.

        Returns:
            Plotly Figure
        """
        if self.analyzer.df.empty:
            return go.Figure()

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=self.analyzer.df['distance_unit'],
            y=self.analyzer.df['pace_per_unit'],
            mode='markers',
            marker=dict(
                size=8,
                color=self.analyzer.df['start_date'].astype(int) / 10**9,  # Convert to timestamp
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Date<br>(newer →)'),
            ),
            text=self.analyzer.df['name'],
            hovertemplate='%{text}<br>Distance: %{x:.2f} %s<br>Pace: %{y:.2f} min/%s<extra></extra>' % (self.unit, self.unit)
        ))

        # Add trendline
        z = np.polyfit(self.analyzer.df['distance_unit'], self.analyzer.df['pace_per_unit'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.analyzer.df['distance_unit'].min(), self.analyzer.df['distance_unit'].max(), 100)

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Pace vs Distance Analysis',
            xaxis_title=f'Distance ({self.unit})',
            yaxis_title=f'Pace (min/{self.unit})',
            height=600,
            hovermode='closest'
        )

        fig.update_yaxis(autorange='reversed')  # Lower pace is better

        return fig

    def create_comparison_chart(self, comparison: Dict) -> go.Figure:
        """
        Create comparison chart between two periods.

        Args:
            comparison: Dictionary from analyzer.compare_periods()

        Returns:
            Plotly Figure with comparison
        """
        if not comparison or 'period1' not in comparison:
            return go.Figure()

        p1 = comparison['period1']['stats']
        p2 = comparison['period2']['stats']

        if not p1 or not p2:
            return go.Figure()

        # Prepare data
        metrics = list(p1.keys())
        period1_values = [p1[m] for m in metrics]
        period2_values = [p2[m] for m in metrics]

        # Create grouped bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name=f"Period 1: {comparison['period1']['start']} to {comparison['period1']['end']}",
            x=metrics,
            y=period1_values,
            marker=dict(color='lightblue')
        ))

        fig.add_trace(go.Bar(
            name=f"Period 2: {comparison['period2']['start']} to {comparison['period2']['end']}",
            x=metrics,
            y=period2_values,
            marker=dict(color='coral')
        ))

        fig.update_layout(
            title='Training Period Comparison',
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            height=600,
            xaxis_tickangle=-45
        )

        return fig
