

import json
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataVisualizer:

    def __init__(self, json_file: str = "logs/driving_data.json",
                 csv_file: str = "logs/driving_data.csv"):
        self.json_file = json_file
        self.csv_file = csv_file
        self.data = None
        self.df = None
        self.stats = {}

    def load_data(self, file_format: str = "both") -> bool:
        try:
            if file_format in ["json", "both"]:
                if os.path.exists(self.json_file):
                    with open(self.json_file, 'r') as f:
                        self.data = json.load(f)
                    print(f"Loaded {len(self.data)} entries from JSON")
                else:
                    print(f"JSON file not found: {self.json_file}")

            if file_format in ["csv", "both"]:
                if os.path.exists(self.csv_file):
                    self.df = pd.read_csv(self.csv_file)
                    print(f"Loaded {len(self.df)} entries from CSV")
                else:
                    print(f"CSV file not found: {self.csv_file}")

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def convert_json_to_csv(self) -> bool:
        try:
            if not self.data:
                print("No JSON data to convert")
                return False

            df_data = []
            for entry in self.data:
                flat_entry = {
                    'timestamp': entry['timestamp'],
                    'car_position_x': entry['state'][0],
                    'car_position_y': entry['state'][1],
                    'angle': np.degrees(np.arctan2(entry['state'][4], entry['state'][3])),
                    'speed': entry['speed'],
                    'sensor_distances': json.dumps(entry['state'][6:]),
                    'traffic_light_state': 'unknown',
                    'action_taken': json.dumps(entry['action']),
                    'reward': entry['reward'],
                    'ai_mode': 'hybrid'  
                }

                if 'traffic_light_state' in entry:
                    flat_entry['traffic_light_state'] = entry['traffic_light_state']
                if 'ai_mode' in entry:
                    flat_entry['ai_mode'] = entry['ai_mode']

                df_data.append(flat_entry)

            df = pd.DataFrame(df_data)

            os.makedirs("logs", exist_ok=True)

            df.to_csv(self.csv_file, index=False)
            print(f"Converted {len(df)} entries to CSV: {self.csv_file}")
            return True

        except Exception as e:
            print(f"Error converting JSON to CSV: {e}")
            return False

    def plot_accuracy_over_time(self, window_size: int = 50) -> plt.Figure:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None

        try:
            try:
                valid_timestamps = []
                for ts in self.df['timestamp']:
                    try:
                        if isinstance(ts, (int, float)):
                            valid_timestamps.append(ts)
                        else:
                            valid_timestamps.append(np.nan)
                    except:
                        valid_timestamps.append(np.nan)

                self.df['datetime'] = pd.to_datetime(valid_timestamps, unit='s', errors='coerce')
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s', errors='coerce')

            self.df['accuracy'] = self.df['reward'].rolling(window=window_size).mean()

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.df['datetime'], self.df['accuracy'],
                   label=f'Accuracy (MA {window_size})', color='blue')

            ax.set_title('Model Accuracy Over Time', fontsize=16)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Accuracy (Reward Moving Average)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error plotting accuracy: {e}")
            return None

    def plot_reward_improvement(self) -> plt.Figure:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None

        try:
            try:
                valid_timestamps = []
                for ts in self.df['timestamp']:
                    try:
                        if isinstance(ts, (int, float)):
                            valid_timestamps.append(ts)
                        else:
                            valid_timestamps.append(np.nan)
                    except:
                        valid_timestamps.append(np.nan)

                self.df['datetime'] = pd.to_datetime(valid_timestamps, unit='s', errors='coerce')
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s', errors='coerce')

            self.df['cumulative_reward'] = self.df['reward'].cumsum()
            self.df['reward_ma'] = self.df['reward'].rolling(window=20).mean()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            ax1.plot(self.df['datetime'], self.df['cumulative_reward'],
                    label='Cumulative Reward', color='green')
            ax1.set_title('Cumulative Reward Over Time', fontsize=14)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Cumulative Reward')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(self.df['datetime'], self.df['reward_ma'],
                    label='Reward (MA 20)', color='orange')
            ax2.set_title('Reward Moving Average', fontsize=14)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Reward (20-frame MA)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error plotting rewards: {e}")
            return None

    def plot_failure_and_struggle_count(self) -> plt.Figure:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None

        try:
            self.df['is_failure'] = self.df['reward'] < -50  # Severe negative reward
            self.df['is_struggle'] = (self.df['reward'] < -10) & (self.df['reward'] >= -50)

            try:
                valid_timestamps = []
                for ts in self.df['timestamp']:
                    try:
                        if isinstance(ts, (int, float)):
                            valid_timestamps.append(ts)
                        else:
                            valid_timestamps.append(np.nan)
                    except:
                        valid_timestamps.append(np.nan)

                self.df['datetime'] = pd.to_datetime(valid_timestamps, unit='s', errors='coerce')
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s', errors='coerce')

            failure_rate = self.df['is_failure'].rolling(window=100).sum() / 100
            struggle_rate = self.df['is_struggle'].rolling(window=100).sum() / 100

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.df['datetime'], failure_rate,
                   label='Failure Rate (per 100 frames)', color='red')
            ax.plot(self.df['datetime'], struggle_rate,
                   label='Struggle Rate (per 100 frames)', color='orange')

            ax.set_title('Failure and Struggle Events Over Time', fontsize=16)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Event Rate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error plotting failures: {e}")
            return None

    def plot_task_completion_time(self) -> plt.Figure:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None

        try:
            start_x = self.df['car_position_x'].iloc[0]
            start_y = self.df['car_position_y'].iloc[0]

            self.df['distance_from_start'] = np.sqrt(
                (self.df['car_position_x'] - start_x)**2 +
                (self.df['car_position_y'] - start_y)**2
            )

            try:
                valid_timestamps = []
                for ts in self.df['timestamp']:
                    try:
                        if isinstance(ts, (int, float)):
                            valid_timestamps.append(ts)
                        else:
                            valid_timestamps.append(np.nan)
                    except:
                        valid_timestamps.append(np.nan)

                self.df['datetime'] = pd.to_datetime(valid_timestamps, unit='s', errors='coerce')
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s', errors='coerce')

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.df['datetime'], self.df['distance_from_start'],
                   label='Distance from Start', color='purple')

            ax.set_title('Task Completion Progress (Distance Traveled)', fontsize=16)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Distance from Start Position', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error plotting task completion: {e}")
            return None

    def plot_safety_violations(self) -> plt.Figure:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None

        try:
            self.df['safety_violation'] = self.df['reward'] < -20

            try:
                valid_timestamps = []
                for ts in self.df['timestamp']:
                    try:
                        if isinstance(ts, (int, float)):
                            valid_timestamps.append(ts)
                        else:
                            valid_timestamps.append(np.nan)
                    except:
                        valid_timestamps.append(np.nan)

                self.df['datetime'] = pd.to_datetime(valid_timestamps, unit='s', errors='coerce')
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                self.df['datetime'] = pd.to_datetime(self.df['timestamp'], unit='s', errors='coerce')

            # Calculate violation rate
            violation_rate = self.df['safety_violation'].rolling(window=50).sum()

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.df['datetime'], violation_rate,
                   label='Safety Violations (50-frame window)', color='red')

            ax.set_title('Safety Violations Over Time', fontsize=16)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Number of Violations', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error plotting safety violations: {e}")
            return None

    def plot_performance_comparison(self) -> plt.Figure:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None

        try:
            if 'ai_mode' not in self.df.columns:
                print("No AI mode data available for comparison")
                return None

            hybrid_data = self.df[self.df['ai_mode'] == 'hybrid']
            pure_ml_data = self.df[self.df['ai_mode'] == 'pure_ml']

            if len(hybrid_data) == 0 or len(pure_ml_data) == 0:
                print("Insufficient data for both AI modes")
                return None

            # Calculate performance
            hybrid_reward = hybrid_data['reward'].mean()
            pure_ml_reward = pure_ml_data['reward'].mean()

            hybrid_violations = (hybrid_data['reward'] < -20).sum() / len(hybrid_data)
            pure_ml_violations = (pure_ml_data['reward'] < -20).sum() / len(pure_ml_data)

            hybrid_speed = hybrid_data['speed'].mean()
            pure_ml_speed = pure_ml_data['speed'].mean()

            # comparison data
            metrics = ['Average Reward', 'Violation Rate', 'Average Speed']
            hybrid_values = [hybrid_reward, hybrid_violations, hybrid_speed]
            pure_ml_values = [pure_ml_reward, pure_ml_violations, pure_ml_speed]

            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 6))
            rects1 = ax.bar(x - width/2, hybrid_values, width, label='Hybrid AI', color='blue')
            rects2 = ax.bar(x + width/2, pure_ml_values, width, label='Pure ML', color='orange')

            ax.set_title('Performance Comparison: Hybrid AI vs Pure ML', fontsize=16)
            ax.set_xlabel('Performance Metrics', fontsize=12)
            ax.set_ylabel('Values', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)

            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error plotting performance comparison: {e}")
            return None

    def plot_sensor_data_analysis(self) -> plt.Figure:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None

        try:
            sensor_data = []
            for sensors_json in self.df['sensor_distances']:
                try:
                    sensors = json.loads(sensors_json)
                    sensor_data.append(sensors)
                except:
                    continue

            if not sensor_data:
                print("No sensor data available")
                return None

            sensor_array = np.array(sensor_data)
            sensor_means = sensor_array.mean(axis=0)
            sensor_stds = sensor_array.std(axis=0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            sensors = range(len(sensor_means))
            ax1.bar(sensors, sensor_means, yerr=sensor_stds,
                   capsize=5, color='green', alpha=0.7)
            ax1.set_title('Average Sensor Readings with Standard Deviation', fontsize=14)
            ax1.set_xlabel('Sensor Index')
            ax1.set_ylabel('Distance')
            ax1.set_xticks(sensors)
            ax1.set_xticklabels([f'Sensor {i}' for i in sensors])
            ax1.grid(True, alpha=0.3)

            corr_matrix = np.corrcoef(sensor_array, rowvar=False)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2,
                       xticklabels=[f'Sensor {i}' for i in sensors],
                       yticklabels=[f'Sensor {i}' for i in sensors])
            ax2.set_title('Sensor Correlation Matrix', fontsize=14)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error plotting sensor data: {e}")
            return None

    def generate_statistical_report(self) -> Dict[str, any]:
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return {}

        try:
            clean_timestamps = []
            for ts in self.df['timestamp']:
                try:
                    if isinstance(ts, (int, float)):
                        clean_timestamps.append(ts)
                    else:
                        clean_timestamps.append(np.nan)
                except:
                    clean_timestamps.append(np.nan)

            report = {
                'total_entries': len(self.df),
                'time_range': {
                    'start': np.nanmin(clean_timestamps),
                    'end': np.nanmax(clean_timestamps),
                    'duration_seconds': np.nanmax(clean_timestamps) - np.nanmin(clean_timestamps)
                },
                'reward_stats': {
                    'mean': self.df['reward'].mean(),
                    'std': self.df['reward'].std(),
                    'min': self.df['reward'].min(),
                    'max': self.df['reward'].max(),
                    'median': self.df['reward'].median()
                },
                'speed_stats': {
                    'mean': self.df['speed'].mean(),
                    'std': self.df['speed'].std(),
                    'min': self.df['speed'].min(),
                    'max': self.df['speed'].max()
                },
                'failure_stats': {
                    'total_failures': (self.df['reward'] < -50).sum(),
                    'failure_rate': (self.df['reward'] < -50).mean(),
                    'total_struggles': ((self.df['reward'] < -10) & (self.df['reward'] >= -50)).sum(),
                    'struggle_rate': ((self.df['reward'] < -10) & (self.df['reward'] >= -50)).mean()
                }
            }

            if 'ai_mode' in self.df.columns:
                ai_modes = self.df['ai_mode'].unique()
                mode_stats = {}
                for mode in ai_modes:
                    mode_data = self.df[self.df['ai_mode'] == mode]
                    mode_stats[mode] = {
                        'entries': len(mode_data),
                        'avg_reward': mode_data['reward'].mean(),
                        'avg_speed': mode_data['speed'].mean(),
                        'failure_rate': (mode_data['reward'] < -50).mean()
                    }
                report['ai_mode_comparison'] = mode_stats

            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32, np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return float(obj) if isinstance(obj, (np.float64, np.float32, np.float16)) else int(obj)
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.ndarray, pd.Series)):
                    return convert_numpy_types(obj.tolist())
                elif isinstance(obj, (np.generic, np.number)):
                    return float(obj) if hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.floating) else int(obj)
                else:
                    return obj

            return convert_numpy_types(report)

        except Exception as e:
            print(f"Error generating statistical report: {e}")
            return {}

    def plot_all_metrics(self, output_dir: str = "visualizations") -> List[plt.Figure]:
        os.makedirs(output_dir, exist_ok=True)

        figures = []

        # Generate plots
        # Not implemented for now
        '''
        accuracy_fig = self.plot_accuracy_over_time()
        if accuracy_fig:
            accuracy_fig.savefig(f"{output_dir}/accuracy_over_time.png")
            figures.append(accuracy_fig)

        reward_fig = self.plot_reward_improvement()
        if reward_fig:
            reward_fig.savefig(f"{output_dir}/reward_improvement.png")
            figures.append(reward_fig)

        failure_fig = self.plot_failure_and_struggle_count()
        if failure_fig:
            failure_fig.savefig(f"{output_dir}/failure_count.png")
            figures.append(failure_fig)

        completion_fig = self.plot_task_completion_time()
        if completion_fig:
            completion_fig.savefig(f"{output_dir}/task_completion.png")
            figures.append(completion_fig)

        safety_fig = self.plot_safety_violations()
        if safety_fig:
            safety_fig.savefig(f"{output_dir}/safety_violations.png")
            figures.append(safety_fig)
        '''
            

        comparison_fig = self.plot_performance_comparison()
        if comparison_fig:
            comparison_fig.savefig(f"{output_dir}/performance_comparison.png")
            figures.append(comparison_fig)

        sensor_fig = self.plot_sensor_data_analysis()
        if sensor_fig:
            sensor_fig.savefig(f"{output_dir}/sensor_analysis.png")
            figures.append(sensor_fig)

        print(f"Generated {len(figures)} plots in {output_dir}/")
        return figures

    def show_interactive_dashboard(self):
        figures = self.plot_all_metrics()
        if figures:
            plt.show()

def main():
    print("=== Data Visualization Module ===")
    visualizer = DataVisualizer()
    print("Loading data...")
    if not visualizer.load_data("both"):
        print("Failed to load data. Creating sample data for demonstration.")
        sample_data = []
        for i in range(100):
            sample_data.append({
                'timestamp': time.time() + i,
                'state': [0.1 + i*0.01, 0.5, 0.3, 1.0, 0.0, 0.5] + [1.0] * 7,
                'action': {'accelerate': True},
                'speed': 2.0 + i*0.01,
                'reward': 5.0 + i*0.1,
                'ai_mode': 'hybrid' if i % 2 == 0 else 'pure_ml'
            })

        os.makedirs("logs", exist_ok=True)
        with open("logs/driving_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)

        visualizer.load_data("json")
        visualizer.convert_json_to_csv()
        visualizer.load_data("both")

    print("\nGenerating statistical report...")
    report = visualizer.generate_statistical_report()
    print(json.dumps(report, indent=2))

    print("\nShowing interactive dashboard...")
    visualizer.show_interactive_dashboard()

if __name__ == "__main__":
    import time
    main()