import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from collections import Counter
import random
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

class LotteryAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with a lottery CSV file."""
        self.df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(self.df)} rows")
        
        # Display column names to understand the data structure
        print("Columns in the dataset:", self.df.columns.tolist())
        
        # Try to automatically identify the draw date and numbers columns
        self.identify_columns()
        
        # Initialize storage for analysis results
        self.results = {
            "descriptive_stats": {},
            "probabilities": {},
            "time_series": {},
            "combinations": {},
            "advanced_models": {},
            "anomalies": {},
            "final_predictions": []
        }
    def analyze_extra_ball(self):
        """Separate analysis for the extra ball"""
        if 'Extra Ball' not in self.df.columns:
            return

        extra_balls = self.df['Extra Ball'].dropna().astype(int)
        counter = Counter(extra_balls)
        
        self.results['extra_ball'] = {
            'frequency': dict(counter),
            'most_common': counter.most_common(10),
            'probability': {num: count/len(extra_balls) for num, count in counter.items()}
        }
        
        # Visualize extra ball distribution
        plt.figure(figsize=(10, 6))
        plt.bar(counter.keys(), counter.values())
        plt.title('Extra Ball Frequency Distribution')
        plt.xlabel('Number')
        plt.ylabel('Count')
        plt.savefig('extra_ball_distribution.png')
        
        return self.results['extra_ball']
    
    def identify_columns(self):
        """Improved column identification"""
        # First try to find explicit date columns
        date_cols = [col for col in self.df.columns 
                    if any(key in col.lower() for key in ['date', 'draw date'])]
        
        if not date_cols:
            # Fallback to timestamp detection
            date_cols = [col for col in self.df.columns 
                        if pd.api.types.is_datetime64_any_dtype(self.df[col])]
        
        if date_cols:
            self.date_column = date_cols[0]
            print(f"Identified date column as: {self.date_column}")
            try:
                self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
            except Exception as e:
                print(f"Date conversion error: {e}")

        # Identify number columns more precisely
        self.number_columns = [col for col in self.df.columns 
                             if any(key in col.lower() for key in ['ball', 'extra'])
                             and col != self.date_column]
        
        if not self.number_columns:
            self.number_columns = self.df.select_dtypes(include=['number']).columns.tolist()
            self.number_columns = [col for col in self.number_columns 
                                 if col != self.date_column]

        print(f"Identified number columns as: {self.number_columns}")
    
    def set_columns(self, date_column=None, number_columns=None):
        """Manually set the date and number columns."""
        if date_column:
            self.date_column = date_column
            try:
                self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
            except:
                print("Warning: Could not convert date column to datetime format")
        
        if number_columns:
            self.number_columns = number_columns
    
    def get_all_numbers(self):
        """Extract all numbers from the dataset into a flat list."""
        all_numbers = []
        for col in self.number_columns:
            all_numbers.extend(self.df[col].dropna().astype(int).tolist())
        return all_numbers
    
    def descriptive_statistics(self):
        """Calculate descriptive statistics for the lottery numbers."""
        all_numbers = self.get_all_numbers()
        
        # Basic statistics
        self.results["descriptive_stats"]["mean"] = np.mean(all_numbers)
        self.results["descriptive_stats"]["median"] = np.median(all_numbers)
        self.results["descriptive_stats"]["mode"] = stats.mode(all_numbers, keepdims=False).mode
        self.results["descriptive_stats"]["variance"] = np.var(all_numbers)
        self.results["descriptive_stats"]["std_dev"] = np.std(all_numbers)
        
        # Frequency distribution
        counter = Counter(all_numbers)
        max_number = max(all_numbers)
        
        # Create a frequency table for all possible numbers
        freq_table = {num: counter.get(num, 0) for num in range(1, max_number + 1)}
        self.results["descriptive_stats"]["frequency"] = freq_table
        
        # Most and least common numbers
        sorted_freq = sorted(freq_table.items(), key=lambda x: x[1], reverse=True)
        self.results["descriptive_stats"]["most_common"] = sorted_freq[:10]
        self.results["descriptive_stats"]["least_common"] = sorted_freq[-10:]
        
        # Visualize frequency distribution
        plt.figure(figsize=(15, 8))
        bars = plt.bar(freq_table.keys(), freq_table.values())
        
        # Color coding: green for most common, red for least common
        most_common_nums = [x[0] for x in sorted_freq[:10]]
        least_common_nums = [x[0] for x in sorted_freq[-10:]]
        
        for i, num in enumerate(freq_table.keys()):
            if num in most_common_nums:
                bars[i-1].set_color('green')
            elif num in least_common_nums:
                bars[i-1].set_color('red')
        
        plt.title('Frequency Distribution of Lottery Numbers')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('frequency_distribution.png')
        
        print("Completed descriptive statistics analysis")
        return self.results["descriptive_stats"]
    
    def probability_analysis(self):
        """Analyze probabilities and patterns in the lottery numbers."""
        all_numbers = self.get_all_numbers()
        max_number = max(all_numbers)
        total_draws = len(self.df)
        numbers_per_draw = len(self.number_columns)
        
        # Probability of each number
        counter = Counter(all_numbers)
        total_numbers = len(all_numbers)
        
        probabilities = {num: counter.get(num, 0) / total_numbers for num in range(1, max_number + 1)}
        self.results["probabilities"]["individual"] = probabilities
        
        # Hot and cold numbers
        sorted_prob = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        hot_threshold = np.percentile(list(probabilities.values()), 75)
        cold_threshold = np.percentile(list(probabilities.values()), 25)
        
        hot_numbers = [num for num, prob in probabilities.items() if prob >= hot_threshold]
        cold_numbers = [num for num, prob in probabilities.items() if prob <= cold_threshold]
        
        self.results["probabilities"]["hot_numbers"] = hot_numbers
        self.results["probabilities"]["cold_numbers"] = cold_numbers
        
        # Visualize hot and cold numbers
        plt.figure(figsize=(15, 8))
        plt.bar(probabilities.keys(), probabilities.values(), color='lightblue')
        
        # Highlight hot numbers in red
        for num in hot_numbers:
            plt.bar(num, probabilities[num], color='red')
        
        # Highlight cold numbers in blue
        for num in cold_numbers:
            plt.bar(num, probabilities[num], color='blue')
        
        plt.title('Probability Distribution of Lottery Numbers')
        plt.xlabel('Number')
        plt.ylabel('Probability')
        plt.axhline(y=1/max_number, color='green', linestyle='--', label='Uniform Probability')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('probability_distribution.png')
        
        # Identify repeating sequences (pairs and triplets)
        if len(self.number_columns) >= 2:
            # Look at ordered pairs within each draw
            pairs = []
            for _, row in self.df.iterrows():
                numbers = [row[col] for col in self.number_columns if pd.notna(row[col])]
                numbers = [int(n) for n in numbers]  # Convert to integers
                numbers.sort()  # Sort to handle different orders
                
                # Extract all possible pairs
                for i in range(len(numbers)):
                    for j in range(i+1, len(numbers)):
                        pairs.append((numbers[i], numbers[j]))
            
            pair_counter = Counter(pairs)
            self.results["probabilities"]["common_pairs"] = pair_counter.most_common(10)
            
            # Triplets if we have enough numbers
            if len(self.number_columns) >= 3:
                triplets = []
                for _, row in self.df.iterrows():
                    numbers = [row[col] for col in self.number_columns if pd.notna(row[col])]
                    numbers = [int(n) for n in numbers]  # Convert to integers
                    numbers.sort()  # Sort to handle different orders
                    
                    # Extract all possible triplets
                    for i in range(len(numbers)):
                        for j in range(i+1, len(numbers)):
                            for k in range(j+1, len(numbers)):
                                triplets.append((numbers[i], numbers[j], numbers[k]))
                
                triplet_counter = Counter(triplets)
                self.results["probabilities"]["common_triplets"] = triplet_counter.most_common(10)
        
        print("Completed probability and pattern analysis")
        return self.results["probabilities"]
    
    def time_series_analysis(self):
        """Analyze time-based patterns in the lottery numbers."""
        if not self.date_column:
            print("Warning: Cannot perform time series analysis without a date column")
            return {}
        
        # Ensure data is sorted by date
        self.df = self.df.sort_values(by=self.date_column)
        
        # Create a time series for each number
        max_number = max(self.get_all_numbers())
        time_series = {num: [] for num in range(1, max_number + 1)}
        
        for _, row in self.df.iterrows():
            date = row[self.date_column]
            drawn_numbers = [int(row[col]) for col in self.number_columns if pd.notna(row[col])]
            
            # For each possible number, mark 1 if drawn, 0 if not
            for num in range(1, max_number + 1):
                time_series[num].append(1 if num in drawn_numbers else 0)
        
        # Convert to pandas series with date index
        dates = self.df[self.date_column].tolist()
        ts_df = pd.DataFrame(time_series, index=dates)
        
        # Calculate moving averages for each number
        window_sizes = [5, 10, 20]
        moving_avgs = {}
        
        for window in window_sizes:
            moving_avgs[window] = ts_df.rolling(window=window).mean()
        
        self.results["time_series"]["moving_averages"] = moving_avgs
        
        # Detect trends and seasonality for frequently drawn numbers
        top_numbers = [num for num, _ in self.results["descriptive_stats"]["most_common"][:5]]
        
        # Try to detect seasonality and trends for top numbers
        trends = {}
        for num in top_numbers:
            try:
                # Resample to monthly frequency for better seasonality detection
                monthly_series = ts_df[num].resample('M').mean()
                
                # Decompose the time series
                decomposition = seasonal_decompose(monthly_series, model='additive', period=12)
                
                trends[num] = {
                    "trend": decomposition.trend.dropna().tolist(),
                    "seasonal": decomposition.seasonal.dropna().tolist(),
                    "residual": decomposition.resid.dropna().tolist()
                }
                
                # Plot the decomposition
                plt.figure(figsize=(14, 10))
                plt.subplot(411)
                plt.plot(decomposition.observed)
                plt.title(f'Time Series Decomposition for Number {num}')
                plt.subplot(412)
                plt.plot(decomposition.trend)
                plt.title('Trend')
                plt.subplot(413)
                plt.plot(decomposition.seasonal)
                plt.title('Seasonality')
                plt.subplot(414)
                plt.plot(decomposition.resid)
                plt.title('Residuals')
                plt.tight_layout()
                plt.savefig(f'time_series_decomposition_{num}.png')
                
            except Exception as e:
                print(f"Could not perform time series decomposition for number {num}: {e}")
        
        self.results["time_series"]["decomposition"] = trends
        
        # Plot autocorrelation and partial autocorrelation for top numbers
        for num in top_numbers:
            try:
                plt.figure(figsize=(12, 8))
                plt.subplot(211)
                plot_acf(ts_df[num], lags=30, ax=plt.gca())
                plt.title(f'Autocorrelation for Number {num}')
                
                plt.subplot(212)
                plot_pacf(ts_df[num], lags=30, ax=plt.gca())
                plt.title(f'Partial Autocorrelation for Number {num}')
                
                plt.tight_layout()
                plt.savefig(f'autocorrelation_{num}.png')
            except Exception as e:
                print(f"Could not plot autocorrelation for number {num}: {e}")
        
        print("Completed time series analysis")
        return self.results["time_series"]
    
    def combinatorial_analysis(self):
        """Analyze combinations and patterns in winning sets."""
        draws = []
        
        # Collect all draws as sorted tuples
        for _, row in self.df.iterrows():
            numbers = [int(row[col]) for col in self.number_columns if pd.notna(row[col])]
            draws.append(tuple(sorted(numbers)))
        
        # Count frequency of each combination
        combination_counter = Counter(draws)
        self.results["combinations"]["most_common"] = combination_counter.most_common(5)
        
        # Check for common number ranges in winning sets
        ranges = []
        for draw in draws:
            min_num = min(draw)
            max_num = max(draw)
            range_size = max_num - min_num
            ranges.append(range_size)
        
        self.results["combinations"]["range_stats"] = {
            "mean": np.mean(ranges),
            "median": np.median(ranges),
            "min": min(ranges),
            "max": max(ranges)
        }
        
        # Visualize the distribution of ranges
        plt.figure(figsize=(10, 6))
        plt.hist(ranges, bins=20, alpha=0.7, color='blue')
        plt.title('Distribution of Number Ranges in Winning Sets')
        plt.xlabel('Range (Max - Min)')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig('range_distribution.png')
        
        # Analyze even-odd distribution
        even_odd_ratios = []
        for draw in draws:
            even_count = sum(1 for num in draw if num % 2 == 0)
            odd_count = len(draw) - even_count
            even_odd_ratios.append((even_count, odd_count))
        
        ratio_counter = Counter(even_odd_ratios)
        self.results["combinations"]["even_odd_distribution"] = {
            "ratios": ratio_counter.most_common(),
            "most_common": ratio_counter.most_common(1)[0]
        }
        
        # Visualize even-odd distribution
        labels = [f"{even}E-{odd}O" for even, odd in ratio_counter.keys()]
        values = list(ratio_counter.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='purple')
        plt.title('Even-Odd Distribution in Winning Sets')
        plt.xlabel('Even-Odd Ratio')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('even_odd_distribution.png')
        
        # Sum analysis
        sums = [sum(draw) for draw in draws]
        self.results["combinations"]["sum_stats"] = {
            "mean": np.mean(sums),
            "median": np.median(sums),
            "min": min(sums),
            "max": max(sums),
            "std_dev": np.std(sums)
        }
        
        # Visualize sum distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sums, bins=20, alpha=0.7, color='green')
        plt.title('Distribution of Sums in Winning Sets')
        plt.xlabel('Sum of Numbers')
        plt.ylabel('Frequency')
        plt.axvline(x=np.mean(sums), color='red', linestyle='--', label=f'Mean: {np.mean(sums):.1f}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('sum_distribution.png')
        
        print("Completed combinatorial analysis")
        return self.results["combinations"]
    
    def advanced_statistical_modeling(self):
        """Apply advanced statistical models to the lottery data."""
        # Extract all draws as lists
        draws = []
        for _, row in self.df.iterrows():
            numbers = [int(row[col]) for col in self.number_columns if pd.notna(row[col])]
            draws.append(sorted(numbers))
        
        # Markov Chain Analysis
        # Simplified: Look at transitions between consecutive draws
        transitions = {}
        for i in range(len(draws) - 1):
            current = tuple(draws[i])
            next_draw = tuple(draws[i + 1])
            
            # Count how many numbers from the current draw appear in the next draw
            common_count = len(set(current).intersection(set(next_draw)))
            
            if common_count not in transitions:
                transitions[common_count] = 0
            transitions[common_count] += 1
        
        # Convert to probabilities
        total = sum(transitions.values())
        transition_probs = {count: freq / total for count, freq in transitions.items()}
        self.results["advanced_models"]["markov_transitions"] = transition_probs
        
        # Visualize transition probabilities
        plt.figure(figsize=(10, 6))
        plt.bar(transition_probs.keys(), transition_probs.values(), color='teal')
        plt.title('Probability of Common Numbers Between Consecutive Draws')
        plt.xlabel('Number of Common Numbers')
        plt.ylabel('Probability')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('markov_transitions.png')
        
        # Cluster Analysis on Draws
        # Convert draws to feature vectors based on properties
        features = []
        for draw in draws:
            # Features: sum, range, even count, max gap
            sum_val = sum(draw)
            range_val = max(draw) - min(draw)
            even_count = sum(1 for num in draw if num % 2 == 0)
            
            # Calculate gaps between consecutive numbers
            gaps = [draw[i+1] - draw[i] for i in range(len(draw)-1)]
            max_gap = max(gaps) if gaps else 0
            
            features.append([sum_val, range_val, even_count, max_gap])
        
        # Apply K-means clustering
        X = np.array(features)
        
        # Determine optimal number of clusters
        inertia = []
        max_clusters = min(10, len(X) - 1)  # Avoid too many clusters
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), inertia, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.grid(alpha=0.3)
        plt.savefig('kmeans_elbow.png')
        
        # Choose elbow point or 3 clusters if unclear
        try:
            # Simple elbow detection
            diffs = np.diff(inertia)
            elbow = np.argmax(np.diff(diffs)) + 1
            optimal_k = min(elbow + 1, max_clusters)
        except:
            optimal_k = min(3, max_clusters)
        
        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Extract cluster centers and map back to meaningful values
        centers = kmeans.cluster_centers_
        
        cluster_profiles = []
        for i, center in enumerate(centers):
            cluster_draws = [draws[j] for j in range(len(draws)) if clusters[j] == i]
            cluster_size = len(cluster_draws)
            
            profile = {
                "cluster_id": i,
                "size": cluster_size,
                "percentage": cluster_size / len(draws) * 100,
                "avg_sum": center[0],
                "avg_range": center[1],
                "avg_even_count": center[2],
                "avg_max_gap": center[3],
                "sample_draws": cluster_draws[:5]
            }
            cluster_profiles.append(profile)
        
        self.results["advanced_models"]["kmeans_clusters"] = cluster_profiles
        
        # Visualize clusters (projection to 2D)
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
        plt.title('K-means Clustering of Lottery Draws')
        plt.xlabel('Sum of Numbers')
        plt.ylabel('Range (Max - Min)')
        plt.colorbar(label='Cluster')
        plt.grid(alpha=0.3)
        plt.savefig('kmeans_clusters.png')
        
        print("Completed advanced statistical modeling")
        return self.results["advanced_models"]
    
    def anomaly_detection(self):
        """Detect anomalies and bias in the lottery data."""
        # Chi-square test for uniform distribution
        all_numbers = self.get_all_numbers()
        max_number = max(all_numbers)
        
        # Count occurrences of each number
        counter = Counter(all_numbers)
        observed = [counter.get(num, 0) for num in range(1, max_number + 1)]
        
        # Expected counts under uniform distribution
        total_numbers = len(all_numbers)
        expected = [total_numbers / max_number] * max_number
        
        # Chi-square test
        chi2, p_value = stats.chisquare(observed, expected)
        
        self.results["anomalies"]["chi_square_test"] = {
            "chi2": chi2,
            "p_value": p_value,
            "significant_bias": p_value < 0.05
        }
        
        # Z-score analysis for outlier frequencies
        frequencies = np.array([counter.get(num, 0) for num in range(1, max_number + 1)])
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        z_scores = (frequencies - mean_freq) / std_freq
        outliers = [(num, z) for num, z in enumerate(z_scores, 1) if abs(z) > 2]
        
        self.results["anomalies"]["frequency_outliers"] = outliers
        
        # Visualize z-scores
        plt.figure(figsize=(15, 8))
        plt.bar(range(1, max_number + 1), z_scores, color=['red' if abs(z) > 2 else 'blue' for z in z_scores])
        plt.axhline(y=2, color='green', linestyle='--', label='Upper Threshold (z=2)')
        plt.axhline(y=-2, color='orange', linestyle='--', label='Lower Threshold (z=-2)')
        plt.title('Z-scores of Number Frequencies')
        plt.xlabel('Number')
        plt.ylabel('Z-score')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('frequency_z_scores.png')
        
        # Runs test for randomness
        if self.date_column:
            # Analyze the sequence of numbers over time
            self.df = self.df.sort_values(by=self.date_column)
            
            # Take a sample of frequently drawn numbers
            top_numbers = [num for num, _ in self.results["descriptive_stats"]["most_common"][:5]]
            
            runs_results = {}
            for num in top_numbers:
                # Create binary sequence: 1 if number drawn, 0 if not
                sequence = []
                for _, row in self.df.iterrows():
                    drawn = [int(row[col]) for col in self.number_columns if pd.notna(row[col])]
                    sequence.append(1 if num in drawn else 0)
                
                # Perform runs test
                try:
                    runs, runs_test_pvalue, _ = stats.wald_wolfowitz(sequence)
                    
                    runs_results[num] = {
                        "runs": runs,
                        "p_value": runs_test_pvalue,
                        "random_sequence": runs_test_pvalue >= 0.05
                    }
                except:
                    # If runs test fails, note it
                    runs_results[num] = "Test failed"
            
            self.results["anomalies"]["runs_test"] = runs_results
        
        print("Completed anomaly detection")
        return self.results["anomalies"]
    
    def generate_predictions(self):
        """Generate optimal number sets based on all previous analyses."""
        all_numbers = self.get_all_numbers()
        max_number = max(all_numbers)
        numbers_per_draw = len(self.number_columns)
        
        # Create a weighted probability distribution based on all analyses
        weights = np.ones(max_number)  # Start with uniform weights
        
        # Factor 1: Historical frequency
        if "frequency" in self.results["descriptive_stats"]:
            freq = self.results["descriptive_stats"]["frequency"]
            freq_weights = np.array([freq.get(num, 0) for num in range(1, max_number + 1)])
            # Normalize
            freq_weights = freq_weights / np.sum(freq_weights) if np.sum(freq_weights) > 0 else freq_weights
            weights *= (0.7 + 0.3 * freq_weights)  # Blend with uniform (70% history, 30% uniform)
        
        # Factor 2: Recent trends (give more weight to hot numbers)
        if "hot_numbers" in self.results["probabilities"]:
            hot_mask = np.zeros(max_number)
            for num in self.results["probabilities"]["hot_numbers"]:
                if 1 <= num <= max_number:
                    hot_mask[num-1] = 1
            weights *= (0.8 + 0.4 * hot_mask)  # Boost hot numbers by 40%
        
        # Factor 3: Include some cold numbers for balance
        if "cold_numbers" in self.results["probabilities"]:
            cold_mask = np.zeros(max_number)
            for num in self.results["probabilities"]["cold_numbers"]:
                if 1 <= num <= max_number:
                    cold_mask[num-1] = 1
            weights *= (0.9 + 0.2 * cold_mask)  # Slight boost to cold numbers
        
        # Factor 4: Avoid outliers that have abnormal frequencies
        if "frequency_outliers" in self.results["anomalies"]:
            outlier_mask = np.ones(max_number)
            for num, z_score in self.results["anomalies"]["frequency_outliers"]:
                if 1 <= num <= max_number:
                    if z_score > 3:  # Extremely high frequency
                        outlier_mask[num-1] = 0.7  # Reduce weight
                    elif z_score < -3:  # Extremely low frequency
                        outlier_mask[num-1] = 0.7  # Reduce weight
            weights *= outlier_mask
        
        # Normalize weights to create a probability distribution
        weights = weights / np.sum(weights)
        
        # Generate 5 sets of numbers
        predictions = []
        for _ in range(5):
            # Sample exactly 6 numbers
            selected = np.random.choice(
                range(1, max_number + 1), 
                size=6,  # Fixed to 6 numbers
                replace=False, 
                p=weights
            )
            selected.sort()
            predictions.append(selected.tolist())
        
        self.results["final_predictions"] = predictions
        
        # Create a visual representation of the final probability distribution
        plt.figure(figsize=(15, 8))
        plt.bar(range(1, max_number + 1), weights, color='purple', alpha=0.7)
        plt.title('Final Weighted Probability Distribution for Number Selection')
        plt.xlabel('Number')
        plt.ylabel('Relative Probability')
        plt.grid(alpha=0.3)
        plt.savefig('final_probability_distribution.png')
        
        # Highlight the selected numbers in each set
        fig, axes = plt.subplots(5, 1, figsize=(15, 20))
        for i, prediction in enumerate(predictions):
            axes[i].bar(range(1, max_number + 1), weights, color='lightgray', alpha=0.5)
            
            # Highlight selected numbers
            for num in prediction:
                axes[i].bar(num, weights[num-1], color='red', alpha=1.0)
            
            axes[i].set_title(f'Prediction Set {i+1}: {prediction}')
            axes[i].set_xlabel('Number')
            axes[i].set_ylabel('Probability')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_sets.png')
        
        print("Generated final predictions")
        return self.results["final_predictions"]
    
    def run_all_analyses(self):
        """Run all analyses in sequence and generate a comprehensive report."""
        self.descriptive_statistics()
        self.probability_analysis()
        self.time_series_analysis()
        self.combinatorial_analysis()
        self.advanced_statistical_modeling()
        self.anomaly_detection()
        self.analyze_extra_ball()
        self.generate_predictions()
        
        # Generate a text report
        report = [
            "Lottery Analysis Report",
            "=======================",
            f"Analysis performed at: {datetime.datetime.now()}",
            f"Total draws analyzed: {len(self.df)}",
            "\nDescriptive Statistics:",
            f"- Most common number: {self.results['descriptive_stats']['most_common'][0]}",
            f"- Least common number: {self.results['descriptive_stats']['least_common'][-1]}",
            f"- Average number: {self.results['descriptive_stats']['mean']:.2f}",
            
            "\nProbability Analysis:",
            f"- Hot numbers: {sorted(self.results['probabilities']['hot_numbers'])}",
            f"- Cold numbers: {sorted(self.results['probabilities']['cold_numbers'])}",
            
            "\nAnomaly Detection:",
            f"- Chi-square p-value: {self.results['anomalies']['chi_square_test']['p_value']:.4f}",
            f"- Significant bias: {self.results['anomalies']['chi_square_test']['significant_bias']}",
            
            "\nFinal Predictions:"
        ]
        
        for i, prediction in enumerate(self.results['final_predictions'], 1):
            report.append(f"Set {i}: {prediction}")
            
        # Save report to file
        report_str = '\n'.join(report)
        with open('lottery_report.txt', 'w') as f:
            f.write(report_str)
            
        print("\n".join(report))
        print("\nAnalysis complete! Check generated plots and lottery_report.txt")
        return self.results

    def simulate_draws(self, num_simulations=1000):
        """Monte Carlo simulation of future draws"""
        all_numbers = self.get_all_numbers()
        max_num = max(all_numbers)
        simulations = []
        
        for _ in range(num_simulations):
            draw = np.random.choice(range(1, max_num+1), 
                                  size=len(self.number_columns),
                                  p=self.results['probabilities']['individual'].values(),
                                  replace=False)
            simulations.append(sorted(draw))
        
        # Analyze simulation results
        self.results['simulations'] = {
            'most_common_simulated': Counter(tuple(d) for d in simulations).most_common(10)
        }
        return self.results['simulations']

    def validate_model(self, test_data):
        """Validate predictions against actual historical data"""
        # Implementation would depend on data format
        pass

    # In your instantiation code:
if __name__ == "__main__":
    analyzer = LotteryAnalyzer('C:/Users/wmmb/OneDrive/Desktop/bootcamp/Lottey_gen.py/Lebanese_Lottery.csv')
    analyzer.set_columns(
        date_column='Date',
        number_columns=[f'Ball {i}' for i in range(1, 7)]  # Only Balls 1-6
    )
    analyzer.run_all_analyses()