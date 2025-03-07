# Lebanese Lottery Number Analyzer

## Overview
This project analyzes Lebanese lottery results using statistical and machine learning techniques to identify patterns, trends, and potential number predictions. The analysis includes frequency distributions, probability calculations, time series trends, and anomaly detection.

## Features
- **Descriptive Statistics**: Calculates mean, median, mode, and frequency distribution of drawn numbers.
- **Probability Analysis**: Identifies hot and cold numbers based on historical data.
- **Time Series Analysis**: Detects trends and seasonality in lottery draws.
- **Combinatorial Analysis**: Examines common number pairs, triplets, and even-odd distributions.
- **Anomaly Detection**: Uses statistical tests to check for biases in number draws.
- **Machine Learning Models**: Implements clustering and Markov Chain analysis.
- **Prediction Generation**: Suggests number combinations based on data-driven insights.

## Project Structure
```
LEB Lottery Number Gen/
├── Data/
│   ├── Lebanese_Lottery.csv  # Dataset of past draws
├── Source/
│   ├── lotto.py              # Main analysis script
├── requirements.txt          # Dependencies list
├── README.md                 # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lebanese-lottery-analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python Source/lotto.py
   ```

## Results
- Generates statistical summaries and visualizations.
- Saves reports and predictions in text and image formats.

## Notes
- This project is for analytical and educational purposes only.
- The lottery is inherently random, and predictions do not guarantee results.

## License
MIT License

