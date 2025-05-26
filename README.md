## F1 Lando Norris vs. Oscar Piastri Performance Comparison.
Telemetry and ML-based drivers' performance comparison using FastF1 and Python. 

**Abu Dhabi Grand Prix 2024**  
*Author: Fiorella Pellerito*  
*Tools: FastF1 · Python · Pandas · Matplotlib · Scikit-Learn*

## Project Overview

This project analyzes and compares the driving performance of McLaren F1 drivers **Lando Norris** and **Oscar Piastri** during the 2024 Abu Dhabi Grand Prix. Using official telemetry and lap data accessed via the `FastF1` Python library, the analysis focuses on:

- Lap-by-lap performance comparison.  
- Throttle and brake telemetry on Lap 28.  
- Race pace consistency (standard deviation). 
- Telemetry-based visual insights.
- A ML model that predicts lap times.

The goal is to simulate the type of post-race analysis used by race engineers and strategists to understand driver behavior, identify performance differences, and evaluate consistency across a Grand Prix.

## Key Insights

- Norris showed stronger average pace over clean laps.  
- Piastri demonstrated slightly more consistent throttle/brake control during Lap 28.  
- Telemetry visualizations reveal distinct braking zones and throttle reapplication patterns between both drivers.

## Technologies Used

- `FastF1` – for accessing official Formula 1 telemetry and timing data.  
- `Pandas` – for data analysis and processing.  
- `Matplotlib` – for visualizing lap time trends and telemetry inputs.  
- `matplotlib.backends.backend_pdf` – for generating a multi-page PDF report.
- `scikit-learn` – for handling predictive analysis (ML Model).

## Outputs

- Lap time comparison chart.  
- Throttle and brake overlay plot for Lap 28. 
- Statistical summary of race pace and consistency.
- Branded multi-page PDF report with McLaren-styled visuals.

## Future Work

- Add corner-specific delta comparisons.  
- Build a Streamlit app version.  
- Extend to other drivers and races (Monaco, Silverstone, etc.).

DISLCAIMER: McLaren Logo was used for illustrative/personal purposes only.
