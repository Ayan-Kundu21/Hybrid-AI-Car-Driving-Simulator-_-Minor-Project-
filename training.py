import numpy as np
import pandas as pd
import threading
import time
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Callable
from ai_module import AIModule
import joblib

class DataLogger:

    def __init__(self, json_file: str = "driving_data.json", csv_file: str = "driving_data.csv"):
        self.json_file = json_file
        self.csv_file = csv_file
        self.data_buffer = []
        self.buffer_size = 1000
        self.lock = threading.Lock()
        self.checkpoint_history = []
        self.max_checkpoints = 10

        # Create logs , if doesn't exist
        os.makedirs("logs", exist_ok=True)
        self.json_file = os.path.join("logs", json_file)
        self.csv_file = os.path.join("logs", csv_file)
        
    def log_state(self, state: np.ndarray, action: Dict[str, bool],
                  speed: float, reward: float, timestamp: float = None,
                  traffic_light_state: str = "unknown", ai_mode: str = "hybrid"):
        """Log a single driving state with additional metadata"""
        if timestamp is None:
            timestamp = time.time()

        log_entry = {
            'timestamp': timestamp,
            'state': state.tolist(),
            'action': action,
            'speed': speed,
            'reward': reward,
            'traffic_light_state': traffic_light_state,
            'ai_mode': ai_mode
        }

        with self.lock:
            self.data_buffer.append(log_entry)

            # Flush buffer if full
            if len(self.data_buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to both JSON and CSV files"""
        if not self.data_buffer:
            return

        # Loaddata
        existing_data = []
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        existing_data.extend(self.data_buffer)

        # Save to JSON file
        with open(self.json_file, 'w') as f:
            json.dump(existing_data, f, indent=2)

        # Convert and save to CSV
        self._save_to_csv(existing_data)

        print(f"Flushed {len(self.data_buffer)} entries to JSON and CSV")
        self.data_buffer.clear()

    def _save_to_csv(self, data: List[Dict]):
        """Save data to CSV format"""
        try:
            df_data = []
            for entry in data:
                flat_entry = {
                    'timestamp': entry['timestamp'],
                    'car_position_x': entry['state'][0],
                    'car_position_y': entry['state'][1],
                    'angle': np.degrees(np.arctan2(entry['state'][4], entry['state'][3])),
                    'speed': entry['speed'],
                    'sensor_distances': json.dumps(entry['state'][6:]),
                    'traffic_light_state': entry.get('traffic_light_state', 'unknown'),
                    'action_taken': json.dumps(entry['action']),
                    'reward': entry['reward'],
                    'ai_mode': entry.get('ai_mode', 'hybrid')
                }
                df_data.append(flat_entry)

            df = pd.DataFrame(df_data)
            df.to_csv(self.csv_file, index=False)

        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def create_checkpoint(self, checkpoint_name: str = None) -> bool:
        try:
            data = self.load_data()

            if not data:
                print("No data to checkpoint")
                return False

            checkpoint_entry = {
                'timestamp': time.time(),
                'name': checkpoint_name or f"checkpoint_{len(self.checkpoint_history)}",
                'data': data.copy(),
                'stats': self.get_data_stats()
            }

            self.checkpoint_history.append(checkpoint_entry)

            if len(self.checkpoint_history) > self.max_checkpoints:
                self.checkpoint_history.pop(0)

            print(f"Created checkpoint: {checkpoint_entry['name']}")
            return True

        except Exception as e:
            print(f"Error creating checkpoint: {e}")
            return False

    def rollback_to_checkpoint(self, checkpoint_index: int = -1) -> bool:
        try:
            if not self.checkpoint_history:
                print("No checkpoints available")
                return False

            if checkpoint_index >= len(self.checkpoint_history) or checkpoint_index < -len(self.checkpoint_history):
                print(f"Invalid checkpoint index: {checkpoint_index}")
                return False

            checkpoint = self.checkpoint_history[checkpoint_index]

            # Save checkpoint data as current data
            with open(self.json_file, 'w') as f:
                json.dump(checkpoint['data'], f, indent=2)

            # Update CSV
            self._save_to_csv(checkpoint['data'])

            print(f"Rolled back to checkpoint: {checkpoint['name']}")
            return True

        except Exception as e:
            print(f"Error rolling back: {e}")
            return False

    def list_checkpoints(self) -> List[Dict]:
        return [{
            'index': i,
            'name': cp['name'],
            'timestamp': cp['timestamp'],
            'entries': len(cp['data']),
            'avg_reward': cp['stats'].get('avg_reward', 0)
        } for i, cp in enumerate(self.checkpoint_history)]

    def rollback_to_previous_entries(self, num_entries: int) -> bool:
        try:
            data = self.load_data()

            if len(data) <= num_entries:
                print(f"Cannot rollback: only {len(data)} entries available")
                return False
            rolled_back_data = data[-num_entries:]

            # Save rolled back data
            with open(self.json_file, 'w') as f:
                json.dump(rolled_back_data, f, indent=2)

            # Update CSV
            self._save_to_csv(rolled_back_data)

            print(f"Rolled back to last {num_entries} entries")
            return True

        except Exception as e:
            print(f"Error rolling back to previous entries: {e}")
            return False
    
    def flush(self):
        """Manually flush buffer"""
        with self.lock:
            self._flush_buffer()
    
    def load_data(self) -> List[Dict]:
        """Load all logged data"""
        if not os.path.exists(self.json_file):
            return []

        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def get_data_stats(self) -> Dict[str, any]:
        """Get statistics about logged data"""
        data = self.load_data()

        if not data:
            return {'total_entries': 0}

        try:
            rewards = []
            speeds = []
            valid_timestamps = []
            action_counts = {}

            for entry in data:
                try:
                    if 'reward' in entry and isinstance(entry['reward'], (int, float)):
                        rewards.append(entry['reward'])
                    if 'speed' in entry and isinstance(entry['speed'], (int, float)):
                        speeds.append(entry['speed'])
                    if 'timestamp' in entry and isinstance(entry['timestamp'], (int, float)):
                        valid_timestamps.append(entry['timestamp'])
                    if 'action' in entry and isinstance(entry['action'], dict):
                        for action, active in entry['action'].items():
                            if active:
                                action_counts[action] = action_counts.get(action, 0) + 1
                except Exception as e:
                    print(f"Warning: Skipping corrupted entry: {e}")
                    continue

            if not valid_timestamps:
                return {
                    'total_entries': len(data),
                    'avg_reward': np.mean(rewards) if rewards else 0,
                    'avg_speed': np.mean(speeds) if speeds else 0,
                    'action_distribution': action_counts,
                    'date_range': {'start': 'N/A', 'end': 'N/A'}
                }

            return {
                'total_entries': len(data),
                'avg_reward': np.mean(rewards) if rewards else 0,
                'avg_speed': np.mean(speeds) if speeds else 0,
                'action_distribution': action_counts,
                'date_range': {
                    'start': datetime.fromtimestamp(min(valid_timestamps)).isoformat(),
                    'end': datetime.fromtimestamp(max(valid_timestamps)).isoformat()
                }
            }

        except Exception as e:
            print(f"Error calculating data stats: {e}")
            return {
                'total_entries': len(data),
                'avg_reward': 0,
                'avg_speed': 0,
                'action_distribution': {},
                'date_range': {'start': 'N/A', 'end': 'N/A'}
            }

class ManualTrainer:
    
    def __init__(self, ai_module: AIModule, data_logger: DataLogger):
        self.ai_module = ai_module
        self.data_logger = data_logger
        
    def train_from_logged_data(self, min_samples: int = 100) -> bool:
        print("Loading training data...")
        data = self.data_logger.load_data()

        if len(data) < min_samples:
            print(f"Insufficient data for training. Need at least {min_samples} samples, got {len(data)}")
            return False

        hybrid_data = [entry for entry in data if entry.get('ai_mode') == 'hybrid']
        pure_ml_data = [entry for entry in data if entry.get('ai_mode') == 'pure_ml']

        print(f"Data distribution - Hybrid: {len(hybrid_data)}, Pure ML: {len(pure_ml_data)}")

        if len(pure_ml_data) < min_samples // 2:
            print(f"Generating synthetic pure_ml data to balance the dataset...")
            self._generate_synthetic_data(min_samples - len(pure_ml_data), "pure_ml")
            data = self.data_logger.load_data()
            pure_ml_data = [entry for entry in data if entry.get('ai_mode') == 'pure_ml']

        print(f"Training with {len(data)} samples...")
        
        training_data = []
        for entry in data:
            action_class = 0 
            if entry['action'].get('accelerate', False):
                action_class = 1
            elif entry['action'].get('turn_left', False):
                action_class = 2
            elif entry['action'].get('turn_right', False):
                action_class = 3
            elif entry['action'].get('change_lane_left', False):
                action_class = 4
            elif entry['action'].get('change_lane_right', False):
                action_class = 5
            
            training_sample = {
                'state': entry['state'],
                'action_class': action_class,
                'speed': entry['speed'],
                'reward': entry['reward']
            }
            training_data.append(training_sample)
        
        # Train the models
        success = self.ai_module.train_models(training_data)
        
        if success:
            print("Training completed successfully!")
            # Save the trained models
            self.ai_module.save_models()
            return True
        else:
            print("Training failed!")
            return False
    def _generate_synthetic_data(self, num_samples: int, mode: str = "hybrid"):
        """Generate synthetic training data for balanced datasets"""
        import random
        from datetime import datetime

        print(f"Generating {num_samples} synthetic {mode} samples...")

        synthetic_data = []
        current_time = time.time()

        for i in range(num_samples):
            if i % 10 == 0:  
                state = [
                    random.uniform(0.1, 0.9),  # x position
                    random.uniform(0.1, 0.9),  # y position
                    random.uniform(0.0, 0.5),  # speed (low due to collision)
                    random.choice([-1, 1]),    # direction cos
                    random.choice([-1, 1]),    # direction sin
                    random.uniform(0.0, 1.0),  # lane
                    random.uniform(0.0, 0.3),  # sensor 1 (close)
                    random.uniform(0.0, 0.3),  # sensor 2 (close)
                    random.uniform(0.0, 0.3),  # sensor 3 (close)
                    random.uniform(0.0, 0.3),  # sensor 4 (close)
                    random.uniform(0.0, 0.3),  # sensor 5 (close)
                    random.uniform(0.0, 0.3),  # sensor 6 (close)
                    random.uniform(0.0, 0.3)   # sensor 7 (close)
                ]
                action = {"brake": True}
                reward = random.uniform(-50, -10)
                speed = random.uniform(0.0, 1.0)
            elif i % 7 == 0:  # Traffic violation scenario
                state = [
                    random.uniform(0.1, 0.9),
                    random.uniform(0.1, 0.9),
                    random.uniform(3.0, 5.0),  # high speed
                    random.choice([-1, 1]),
                    random.choice([-1, 1]),
                    random.uniform(0.0, 1.0),
                    random.uniform(0.5, 1.0),  # sensors show clear path
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0)
                ]
                action = {"accelerate": True}
                reward = random.uniform(-20, 5)
                speed = random.uniform(4.0, 5.0)
            else:  # Normal driving scenario
                state = [
                    random.uniform(0.1, 0.9),
                    random.uniform(0.1, 0.9),
                    random.uniform(1.0, 4.0),  # normal speed
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5),
                    random.uniform(0.0, 1.0),
                    random.uniform(0.3, 1.0),  # sensors show safe distance
                    random.uniform(0.3, 1.0),
                    random.uniform(0.3, 1.0),
                    random.uniform(0.3, 1.0),
                    random.uniform(0.3, 1.0),
                    random.uniform(0.3, 1.0),
                    random.uniform(0.3, 1.0)
                ]
                # Random normal action
                actions = ["accelerate", "turn_left", "turn_right", "brake"]
                chosen_action = random.choice(actions)
                action = {chosen_action: True}
                reward = random.uniform(5, 15)
                speed = random.uniform(1.0, 4.0)

            #light info
            traffic_light_state = random.choice(["green", "yellow", "red"])

            synthetic_entry = {
                'timestamp': current_time + i,
                'state': state,
                'action': action,
                'speed': speed,
                'reward': reward,
                'traffic_light_state': traffic_light_state,
                'ai_mode': mode
            }
            synthetic_data.append(synthetic_entry)

        existing_data = self.data_logger.load_data()
        existing_data.extend(synthetic_data)

        with open(self.data_logger.json_file, 'w') as f:
            json.dump(existing_data, f, indent=2)

        self.data_logger._save_to_csv(existing_data)

        print(f"Generated {len(synthetic_data)} synthetic {mode} samples")

    
    def evaluate_model_performance(self) -> Dict[str, float]:
        data = self.data_logger.load_data()
        
        if len(data) < 50:
            return {'error': 'Insufficient data for evaluation'}
        
        recent_data = data[-200:] if len(data) > 200 else data
        
        correct_predictions = 0
        speed_errors = []
        
        for entry in recent_data:
            state = np.array(entry['state'])
            actual_action = entry['action']
            actual_speed = entry['speed']
            
            predicted_action = self.ai_module.make_decision(state)
            predicted_speed = self.ai_module.predict_speed(state)
            
            if self._actions_match(actual_action, predicted_action):
                correct_predictions += 1
            
            speed_errors.append(abs(actual_speed - predicted_speed))
        
        accuracy = correct_predictions / len(recent_data)
        avg_speed_error = np.mean(speed_errors)
        
        return {
            'action_accuracy': accuracy,
            'avg_speed_error': avg_speed_error,
            'evaluation_samples': len(recent_data)
        }
    
    def _actions_match(self, action1: Dict[str, bool], action2: Dict[str, bool]) -> bool:
        # Simple matching - check if main action is the same
        main_actions = ['accelerate', 'brake', 'turn_left', 'turn_right', 
                       'change_lane_left', 'change_lane_right']
        
        for action in main_actions:
            if action1.get(action, False) and action2.get(action, False):
                return True
        
        # If both are "do nothing" (no main action)
        if not any(action1.get(action, False) for action in main_actions) and \
           not any(action2.get(action, False) for action in main_actions):
            return True
        
        return False
    
    def generate_training_report(self) -> str:
        stats = self.data_logger.get_data_stats()
        ai_stats = self.ai_module.get_training_stats()
        performance = self.evaluate_model_performance()
        
        report = f"""
=== TRAINING REPORT ===
Generated: {datetime.now().isoformat()}

DATA STATISTICS:
- Total logged entries: {stats.get('total_entries', 0)}
- Average reward: {stats.get('avg_reward', 0):.3f}
- Average speed: {stats.get('avg_speed', 0):.3f}
- Data collection period: {stats.get('date_range', {}).get('start', 'N/A')} to {stats.get('date_range', {}).get('end', 'N/A')}

ACTION DISTRIBUTION:
"""
        
        for action, count in stats.get('action_distribution', {}).items():
            percentage = (count / stats.get('total_entries', 1)) * 100
            report += f"- {action}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
AI MODEL STATUS:
- Decision classifier trained: {ai_stats['classifier_trained']}
- Speed regressor trained: {ai_stats['regressor_trained']}
- Q-table size: {ai_stats['q_table_size']} states
- Current exploration rate (epsilon): {ai_stats['epsilon']:.3f}

PERFORMANCE METRICS:
"""
        
        if 'error' not in performance:
            report += f"- Action prediction accuracy: {performance['action_accuracy']:.3f}\n"
            report += f"- Average speed prediction error: {performance['avg_speed_error']:.3f}\n"
            report += f"- Evaluation samples: {performance['evaluation_samples']}\n"
        else:
            report += f"- {performance['error']}\n"
        
        report += "\n=== END REPORT ===\n"
        
        return report

class AutomatedTrainer:
    
    def __init__(self, ai_module: AIModule, data_logger: DataLogger):
        self.ai_module = ai_module
        self.data_logger = data_logger
        
        # Training configuration
        self.training_interval = 300  # seconds (5 minutes)
        self.min_new_samples = 50
        self.max_training_samples = 2000
        
        # Threading
        self.training_thread = None
        self.running = False
        self.last_training_time = 0
        self.last_sample_count = 0
        
        # Performance tracking
        self.training_history = []
        
    def start_automated_training(self):
        if self.running:
            print("Automated training is already running!")
            return
        
        self.running = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        print("Automated training started!")
    
    def stop_automated_training(self):
        self.running = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        print("Automated training stopped!")
    
    def _training_loop(self):
        while self.running:
            try:
                current_time = time.time()
                
                if (current_time - self.last_training_time) >= self.training_interval:
                    self._check_and_train()
                    self.last_training_time = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in automated training loop: {e}")
                time.sleep(30)  # Wait 
    
    def _check_and_train(self):
        """Check if training is needed and perform it"""
        data = self.data_logger.load_data()
        current_sample_count = len(data)
        
        new_samples = current_sample_count - self.last_sample_count
        
        if new_samples >= self.min_new_samples:
            print(f"Automated training triggered: {new_samples} new samples")
            
            recent_data = data[-self.max_training_samples:] if len(data) > self.max_training_samples else data
            
            training_data = []
            for entry in recent_data:
                action_class = 0  
                if entry['action'].get('accelerate', False):
                    action_class = 1
                elif entry['action'].get('turn_left', False):
                    action_class = 2
                elif entry['action'].get('turn_right', False):
                    action_class = 3
                elif entry['action'].get('change_lane_left', False):
                    action_class = 4
                elif entry['action'].get('change_lane_right', False):
                    action_class = 5
                
                training_sample = {
                    'state': entry['state'],
                    'action_class': action_class,
                    'speed': entry['speed'],
                    'reward': entry['reward']
                }
                training_data.append(training_sample)
            

            start_time = time.time()
            success = self.ai_module.train_models(training_data)
            training_time = time.time() - start_time
            
            training_event = {
                'timestamp': time.time(),
                'samples_used': len(training_data),
                'new_samples': new_samples,
                'success': success,
                'training_time': training_time
            }
            self.training_history.append(training_event)
            
            if success:
                print(f"Automated training completed in {training_time:.2f}s")
                self.ai_module.save_models()
                self.last_sample_count = current_sample_count
            else:
                print("Automated training failed")
        
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-50:]
    
    def get_training_status(self) -> Dict[str, any]:
        """Get current automated training status"""
        data = self.data_logger.load_data()
        current_sample_count = len(data)
        new_samples = current_sample_count - self.last_sample_count
        
        next_training_in = max(0, self.training_interval - (time.time() - self.last_training_time))
        
        status = {
            'running': self.running,
            'total_samples': current_sample_count,
            'new_samples_since_last_training': new_samples,
            'next_training_in_seconds': next_training_in,
            'training_events': len(self.training_history),
            'last_training_success': self.training_history[-1]['success'] if self.training_history else None
        }
        
        return status
    
    def force_training(self) -> bool:
        print("Forcing immediate training...")
        self._check_and_train()
        return self.training_history[-1]['success'] if self.training_history else False

class TrainingManager:
    
    def __init__(self, ai_module: AIModule):
        self.ai_module = ai_module
        self.data_logger = DataLogger()
        self.manual_trainer = ManualTrainer(ai_module, self.data_logger)
        self.automated_trainer = AutomatedTrainer(ai_module, self.data_logger)
        self.pure_ml_module = None  # Will be initialized when needed
        
    def log_driving_data(self, state: np.ndarray, action: Dict[str, bool],
                         speed: float, reward: float, traffic_light_state: str = "unknown",
                         ai_mode: str = "hybrid"):
        self.data_logger.log_state(state, action, speed, reward,
                                   traffic_light_state, ai_mode)

        if ai_mode == "pure_ml":
            if self.pure_ml_module is None:
                from ai_module import PureMLModule
                self.pure_ml_module = PureMLModule()
            self.pure_ml_module.collect_training_data(state, action, speed, reward)
        else:
            self.ai_module.collect_training_data(state, action, speed, reward)
    
    def start_real_time_training(self):
        self.automated_trainer.start_automated_training()
    
    def stop_real_time_training(self):
        self.automated_trainer.stop_automated_training()
    
    def manual_train(self) -> bool:
        return self.manual_trainer.train_from_logged_data()
    
    def get_comprehensive_status(self) -> Dict[str, any]:
        status = {
            'data_stats': self.data_logger.get_data_stats(),
            'ai_stats': self.ai_module.get_training_stats(),
            'automated_training': self.automated_trainer.get_training_status(),
            'performance': self.manual_trainer.evaluate_model_performance()
        }

        # Add pure ML stats if available
        if self.pure_ml_module:
            status['pure_ml_stats'] = self.pure_ml_module.get_training_stats()

        return status

    def create_checkpoint(self, checkpoint_name: str = None) -> bool:
        return self.data_logger.create_checkpoint(checkpoint_name)

    def rollback_to_checkpoint(self, checkpoint_index: int = -1) -> bool:
        return self.data_logger.rollback_to_checkpoint(checkpoint_index)

    def list_checkpoints(self) -> List[Dict]:
        return self.data_logger.list_checkpoints()

    def rollback_to_previous_entries(self, num_entries: int) -> bool:
        return self.data_logger.rollback_to_previous_entries(num_entries)

    def get_rollback_options(self) -> Dict[str, any]:
        return {
            'checkpoints': self.list_checkpoints(),
            'current_entries': len(self.data_logger.load_data())
        }
    
    def generate_report(self) -> str:
        return self.manual_trainer.generate_training_report()
    
    def cleanup(self):
        self.data_logger.flush()
        self.automated_trainer.stop_automated_training()

if __name__ == "__main__":
    from ai_module import AIModule
    
    ai = AIModule()
    trainer = TrainingManager(ai)
    
    print("Generating test data...")
    for i in range(100):
        state = np.random.random(13)
        action = {'accelerate': True} if i % 2 == 0 else {'brake': True}
        speed = np.random.uniform(0, 5)
        reward = np.random.uniform(-10, 10)
        
        trainer.log_driving_data(state, action, speed, reward)
    
    print("\nTesting manual training...")
    success = trainer.manual_train()
    print(f"Manual training success: {success}")
    print("\nTraining Report:")
    print(trainer.generate_report())
    
    trainer.cleanup()