# Marathon Training Analyzer - Enhanced Edition

A comprehensive Python application for analyzing running training data with support for Strava API, CSV/Excel imports, advanced analytics, and interactive visualizations.

## Features

### 📊 Data Sources
- **Strava API**: Direct integration with Strava for real-time activity data
- **CSV Import**: Load Strava bulk export files or custom CSV formats
- **Excel Import**: Support for .xlsx and .xls files
- **Data Caching**: Save and reload data for faster access
- **Multi-Source Merging**: Combine data from multiple sources with automatic deduplication

### 📈 Advanced Analytics
- **Training Load Metrics**: ATL (Acute Training Load), CTL (Chronic Training Load), TSB (Training Stress Balance)
- **Injury Risk Detection**: Identify rapid mileage increases, insufficient recovery, and high training monotony
- **Heart Rate Zone Analysis**: Analyze time spent in different HR zones with personalized thresholds
- **Multiple Prediction Models**: Marathon time predictions using Riegel, VDOT (Daniels), and Cameron formulas
- **Period Comparison**: Compare training cycles, months, or years side-by-side
- **Long Run Analysis**: Track and analyze long runs with customizable distance thresholds
- **Progression Tracking**: Weekly or monthly aggregation with automatic switching based on data span

### 📉 Visualizations
- **Training Dashboard**: Interactive overview of key metrics
- **Calendar Heatmap**: GitHub-style activity calendar
- **Training Load Charts**: Fitness, fatigue, and freshness over time
- **Heart Rate Zones**: Distribution charts and time analysis
- **Elevation Analysis**: Elevation gain tracking and correlation with pace
- **Pace vs Distance**: Scatter plots with trend analysis
- **Comparison Charts**: Visual period-to-period comparisons

### ⚙️ Additional Features
- **Goal Setting**: Set race goals with target times and track progress
- **Configuration Management**: Persistent user preferences and settings
- **Report Export**: Generate JSON and Markdown reports
- **Data Export**: Export analyzed data to CSV or Excel
- **Unit System Support**: Full metric (km) and imperial (miles) support

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository:
```bash
git clone https://github.com/markgreig/running_analysis.git
cd running_analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python running_analysis.py
```

## Quick Start Guide

### First Run
On first launch, the app will guide you through initial setup:
- Select your preferred unit system (km or miles)
- Set your long run distance threshold
- Optionally configure heart rate settings

### Loading Data

#### Option 1: Strava API
1. Create a Strava API application at https://www.strava.com/settings/api
2. Note your Client ID and Client Secret
3. In the app, select "Load Data" → "Fetch from Strava API"
4. Enter your credentials when prompted
5. Authorize the app in your browser

#### Option 2: CSV File
1. Export your data from Strava:
   - Go to https://www.strava.com/athlete/delete_your_account
   - Click "Request Your Archive"
   - Wait for email with download link
   - Extract the `activities.csv` file
2. In the app, select "Load Data" → "Import from CSV file"
3. Enter the path to your `activities.csv` file

#### Option 3: Excel File
1. Prepare an Excel file with your running data (same format as CSV)
2. Select "Load Data" → "Import from Excel file"
3. Enter the path to your Excel file

### Strava CSV Format
The app expects CSV files with these columns (Strava export format):
- **Activity Date** or **Start Time**: Date and time of activity
- **Activity Type**: Type of activity (filters for "Run")
- **Distance**: Distance in meters (uses second Distance column if duplicates exist)
- **Moving Time**: Moving time in seconds
- **Elapsed Time**: Total elapsed time in seconds
- **Average Heart Rate** (optional): Average HR in bpm
- **Max Heart Rate** (optional): Max HR in bpm
- **Elevation Gain** (optional): Elevation gain in meters
- **Relative Effort** (optional): Strava's training load metric

**Example CSV structure:**
```csv
Activity ID,Activity Date,Activity Name,Activity Type,...,Distance,...,Moving Time,Average Heart Rate,...
123456789,"15 Nov 2025, 13:31:49","Morning Run",Run,...,10000,...,2400,145,...
```

## Usage Guide

### Main Menu Options

1. **Load Data**: Choose from API, CSV, Excel, cache, or merge multiple sources
2. **View Training Summary**: Overview of total runs, distance, pace, etc.
3. **Analyze Progression**: Weekly/monthly trends with visualizations
4. **Analyze Long Runs**: Filter and analyze runs above a distance threshold
5. **View Marathon Predictions**: Time predictions from multiple models
6. **Analyze Training Load**: ATL/CTL/TSB metrics and charts
7. **Analyze Heart Rate Zones**: Time distribution across HR zones
8. **Check Injury Risks**: Identify training risks and get warnings
9. **Compare Training Periods**: Side-by-side comparison of date ranges
10. **Generate Full Report**: Comprehensive report with all metrics
11. **Create Visualizations**: Generate interactive HTML charts
12. **Manage Goals**: Set and track race goals
13. **Export Data**: Export to CSV or Excel
14. **Settings**: Configure preferences and view settings
15. **Exit**: Close the application

### Training Load Interpretation

**ATL (Acute Training Load)**: 7-day rolling average
- Represents recent fatigue
- Higher = more tired

**CTL (Chronic Training Load)**: 42-day rolling average
- Represents fitness
- Higher = more fit

**TSB (Training Stress Balance)**: CTL - ATL
- Represents freshness
- Negative (-10 to -30): Productive training, some fatigue
- Near zero (-10 to +5): Optimal balance
- Positive (+5 to +15): Fresh and ready to race
- Very positive (>+15): Possible detraining

### Heart Rate Zones

The app uses percentage of max HR:
- **Zone 1 (Recovery)**: 50-60% - Easy recovery runs
- **Zone 2 (Aerobic)**: 60-70% - Base building, long runs
- **Zone 3 (Tempo)**: 70-80% - Tempo runs, aerobic endurance
- **Zone 4 (Threshold)**: 80-90% - Lactate threshold training
- **Zone 5 (VO2 Max)**: 90-100% - High-intensity intervals

**Recommended distribution**: 80% easy (Z1-Z2), 20% hard (Z3-Z5)

### Marathon Prediction Models

**Riegel Formula**: Uses race pace with distance-specific fatigue factors
- Most accurate for runners with recent race data
- Adjusts: 5K → 15% slower, 10K → 11%, Half → 6%

**VDOT (Daniels)**: Based on VO2 max estimation
- More scientific approach
- Accounts for fitness level

**Cameron Formula**: Based on longest run pace
- Conservative estimate
- Adds ~5% to long run pace

**Average Prediction**: Mean of all available models with confidence range

## File Structure

```
running_analysis/
├── running_analysis.py    # Main application entry point
├── data_sources.py         # Data loading and import modules
├── analyzers.py            # Analysis engines and algorithms
├── visualizers.py          # Chart and graph generation
├── reporters.py            # Report generation and export
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── activities.csv          # Example data file (if included)
```

## Configuration

Settings are stored in `~/.marathon_analyzer/config.json`

**Configurable options:**
- Unit system (km or miles)
- Long run thresholds
- Maximum and resting heart rate
- Default fetch period
- Cache enabled/disabled
- Auto-save reports
- Training goals

## Tips and Best Practices

### Getting the Most Accurate Predictions
1. Include recent race efforts (within 6 months)
2. Race efforts should be at 5K, 10K, or half marathon distances
3. Ensure GPS accuracy for distance measurements
4. More data = better predictions

### Avoiding Injury
1. Watch for rapid mileage increase warnings (>10% week-over-week)
2. Maintain 80/20 easy/hard training (check HR zones)
3. Include regular rest days
4. Monitor TSB - avoid staying below -30 for extended periods

### Optimizing Training
1. Build base with Zone 2 running (60-70% max HR)
2. Gradually increase CTL (fitness) over 8-12 weeks
3. Taper before races: reduce volume to increase TSB to +5 to +15
4. Compare current cycle to previous successful training blocks

## Troubleshooting

### CSV Import Issues
- **Error: "No running activities found"**: Check that Activity Type column contains "Run"
- **Date parsing errors**: Ensure dates are in format "DD MMM YYYY, HH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
- **Encoding errors**: File will auto-detect encoding, but try UTF-8 if issues persist
- **Distance showing as 0**: Check that you're using the detailed Distance column (second one)

### Strava API Issues
- **Authentication fails**: Verify Client ID and Client Secret are correct
- **No data returned**: Check that you've authorized "activity:read_all" scope
- **Token expired**: App will automatically refresh tokens

### General Issues
- **No heart rate data**: Many features work without HR data; only HR zone analysis requires it
- **Predictions seem off**: Ensure you have recent race efforts; models work best with 5K/10K/half marathon data
- **Charts not opening**: Check that webbrowser module is working; manually open HTML files in browser

## Data Privacy

- **No data is sent to external servers** except Strava API (if used)
- **API credentials are never stored permanently**
- **All data is stored locally** in `~/.marathon_analyzer/`
- **Cache files** are stored as pickled DataFrames for faster loading

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional prediction models
- More visualization types
- Training plan generation
- Workout recommendations
- Mobile app integration

## License

MIT License - feel free to use, modify, and distribute

## Credits

**Analysis Algorithms:**
- Training load concepts: TrainingPeaks, Joe Friel
- VDOT formulas: Jack Daniels' "Running Formula"
- Prediction models: Pete Riegel, various running science sources

**Author**: Enhanced by Claude with modular architecture

## Version History

**v2.0.0 (Enhanced Edition)**
- Added CSV/Excel import support
- Implemented training load metrics (ATL/CTL/TSB)
- Added injury risk detection
- Multiple prediction models
- Heart rate zone analysis
- Calendar heatmap visualization
- Goal tracking
- Configuration management
- Modular code architecture

**v1.0.0**
- Initial release with Strava API support

## Support

For issues, questions, or feature requests:
- Check the Troubleshooting section above
- Review the CSV format requirements
- Ensure all dependencies are installed
- Check Python version (3.8+ required)

---

**Happy Training! 🏃‍♂️💪**
