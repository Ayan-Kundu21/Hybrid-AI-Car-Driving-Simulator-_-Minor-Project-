import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
from typing import Dict, List, Tuple, Optional
import random
from collections import deque

class SafetyModule:
    
    def __init__(self):
        self.min_safe_distance = 100  # Increased for better safety
        self.emergency_brake_distance = 60   # Increased for earlier braking
        self.lane_change_safe_distance = 80  # Safe distance for lane changes
        self.border_warning_distance = 100   # Distance from border to start warning
        self.early_brake_distance = 120      # Distance to start slowing down
        self.swerve_distance = 90            # Distance to consider swerving
        
    def check_collision_risk(self, sensor_readings: List[float], car_y: float = None) -> Dict[str, bool]:
        risks = {
            'front_collision': False,
            'left_collision': False,
            'right_collision': False,
            'emergency_brake': False,
            'obstacle_warning': False,
            'early_brake_warning': False,
            'swerve_left_possible': False,
            'swerve_right_possible': False,
            'road_edge_warning': False,
            'near_top_edge': False,
            'near_bottom_edge': False
        }
        
        # Front sensors (indices 2, 3, 4 for -22.5°, 0°, 22.5°)
        front_sensors = sensor_readings[2:5]
        min_front_distance = min(front_sensors)
        
        if min_front_distance < self.emergency_brake_distance:
            risks['emergency_brake'] = True
            risks['front_collision'] = True
        elif min_front_distance < self.min_safe_distance:
            risks['front_collision'] = True
            risks['obstacle_warning'] = True
        elif min_front_distance < self.early_brake_distance:
            risks['early_brake_warning'] = True
        
        # Left sensors (indices 0, 1) - check for swerving possibilities
        left_sensors = sensor_readings[0:2]
        min_left_distance = min(left_sensors)
        if min_left_distance < self.lane_change_safe_distance:
            risks['left_collision'] = True
        elif min_left_distance > self.swerve_distance:
            risks['swerve_left_possible'] = True
        
        # Right sensors (indices 5, 6) - check for swerving possibilities
        right_sensors = sensor_readings[5:7]
        min_right_distance = min(right_sensors)
        if min_right_distance < self.lane_change_safe_distance:
            risks['right_collision'] = True
        elif min_right_distance > self.swerve_distance:
            risks['swerve_right_possible'] = True
        
        # Check for road edge proximity (top and bottom of road)
        if car_y is not None:
            # Road boundaries: Y=150 (top) to Y=650 (bottom)
            # Car should stay between Y=180 to Y=620 for safety
            if car_y < 200:  # Near top edge
                risks['near_top_edge'] = True
                risks['road_edge_warning'] = True
            elif car_y > 600:  # Near bottom edge
                risks['near_bottom_edge'] = True
                risks['road_edge_warning'] = True
        
        return risks
    
    def traffic_light_compliance(self, traffic_light_state: str, distance_to_light: float) -> Dict[str, bool]:
        """Check traffic light compliance"""
        compliance = {
            'must_stop': False,
            'can_proceed': True,
            'prepare_to_stop': False
        }
        
        if traffic_light_state == "red":
            if distance_to_light < 100:
                compliance['must_stop'] = True
                compliance['can_proceed'] = False
        elif traffic_light_state == "yellow":
            if distance_to_light < 80:
                compliance['prepare_to_stop'] = True
            elif distance_to_light > 120:
                compliance['can_proceed'] = True
        
        return compliance
    
    def lane_keeping_check(self, current_lane: int, target_lane: int, sensor_readings: List[float]) -> bool:
        """Check if lane change is safe"""
        if current_lane == target_lane:
            return True
        
        risks = self.check_collision_risk(sensor_readings)
        
        if target_lane < current_lane:  # Moving left
            return not risks['left_collision']
        else:  # Moving right
            return not risks['right_collision']

class ReinforcementLearning:
    """Q-Learning based reinforcement learning for adaptive behavior"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = {}
        self.memory = deque(maxlen=10000)
        
    def discretize_state(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete representation for Q-table"""
        discretized = []
        
        # Position (coarse grid)
        discretized.append(int(state[0] * 10))  # x position
        discretized.append(int(state[1] * 10))  # y position
        
        # Speed
        discretized.append(int(state[2] * 5))   # speed
        
        # Direction (8 directions)
        angle = np.arctan2(state[4], state[3])
        discretized.append(int((angle + np.pi) / (2 * np.pi) * 8))
        
        # Lane
        discretized.append(int(state[5] * 2))
        
        # Sensor readings (close, medium, far)
        for sensor in state[6:]:
            if sensor < 0.3:
                discretized.append(0)  # close
            elif sensor < 0.7:
                discretized.append(1)  # medium
            else:
                discretized.append(2)  # far
        
        return str(discretized)
    
    def get_action(self, state: np.ndarray) -> int:
        state_key = self.discretize_state(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_key])
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))
    
    def calculate_reward(self, state: np.ndarray, action: int, next_state: np.ndarray, 
                        collision: bool, traffic_violation: bool) -> float:
        reward = 0.0
        
        # Base reward for moving forward
        speed_reward = next_state[2] * 10  # reward for maintaining speed
        reward += speed_reward
        
        # Penalty for collision
        if collision:
            reward -= 100
        
        # Penalty for traffic violations
        if traffic_violation:
            reward -= 50
        
        # Reward for staying in lane
        if abs(next_state[5] - round(next_state[5])) < 0.1:  # close to lane center
            reward += 5
        
        # Penalty for erratic steering
        if action in [1, 2]:  # turning actions
            reward -= 1
        
        # Reward for maintaining safe distance
        sensor_readings = next_state[6:]
        min_sensor = min(sensor_readings)
        if min_sensor > 0.5:  # safe distance
            reward += 2
        elif min_sensor < 0.2:  # too close
            reward -= 10
        
        return reward

class AIModule:
    
    def __init__(self):
        # Random Forest models
        self.decision_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.speed_regressor = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        
        # Safety and RL modules
        self.safety_module = SafetyModule()
        self.rl_module = ReinforcementLearning(state_size=13, action_size=6)  # 6 possible actions

        self.classifier_trained = False
        self.regressor_trained = False

        self.action_map = {
            0: {'accelerate': True},
            1: {'turn_left': True},
            2: {'turn_right': True},
            3: {'brake': True},
            4: {'change_lane_left': True},
            5: {'change_lane_right': True}
        }
        
        # Data collection
        self.training_data = []
        
    def make_decision(self, state: np.ndarray, traffic_light_state: str = "green",
                     distance_to_light: float = 1000) -> Dict[str, bool]:
        """Make driving decision with enhanced collision avoidance and swerving"""
        car_x_normalized = state[0]  
        car_y_normalized = state[1]  
        car_y_actual = car_y_normalized * 800  
        car_speed = state[2] 
        car_direction_cos = state[3]  
        car_direction_sin = state[4]
        current_angle = np.arctan2(car_direction_sin, car_direction_cos) * 180 / np.pi
        
        sensor_readings = state[6:] * 150  
        safety_risks = self.safety_module.check_collision_risk(sensor_readings.tolist(), car_y_actual)
        traffic_compliance = self.safety_module.traffic_light_compliance(
            traffic_light_state, distance_to_light
        )
        
        near_horizontal_boundary = car_x_normalized > 0.95 or car_x_normalized < 0.05
        facing_backwards = abs(current_angle) > 90 
        
        # Emergency brake for collisions and traffic violations
        if safety_risks['emergency_brake'] or traffic_compliance['must_stop']:
            return {'brake': True}
        
        # Road edge avoidance - priority over other actions
        if safety_risks['near_top_edge']:
            if safety_risks['swerve_right_possible']:
                return {'turn_right': True}  # Swerve away from top edge
            else:
                return {'brake': True}
        
        if safety_risks['near_bottom_edge']:
            if safety_risks['swerve_left_possible']:
                return {'turn_left': True}  # Swerve away from bottom edge
            else:
                return {'brake': True}
        
        if safety_risks['front_collision']:
            # Prefer swerving over braking when possible
            if safety_risks['swerve_left_possible'] and not safety_risks['near_top_edge']:
                return {'turn_left': True}  # Swerve left
            elif safety_risks['swerve_right_possible'] and not safety_risks['near_bottom_edge']:
                return {'turn_right': True}  # Swerve right
            else:
                return {'brake': True}  # Brake as last resort
        
        # Early braking for approaching obstacles
        if safety_risks['early_brake_warning']:
            # Slow down early but prefer swerving if possible
            if safety_risks['swerve_left_possible'] and not safety_risks['near_top_edge']:
                return {'turn_left': True}  # Swerve left early
            elif safety_risks['swerve_right_possible'] and not safety_risks['near_bottom_edge']:
                return {'turn_right': True}  # Swerve right early
            elif car_speed > 0.6:  # Only brake if going fast enough
                return {'brake': True}  # Early braking
        
        # Prevent U-turns and backwards driving by discouraging excessive turning
        if facing_backwards:
            if current_angle > 90:
                return {'turn_right': True}  # Turn right to face forward
            elif current_angle < -90:
                return {'turn_left': True}   # Turn left to face forward

        # ENHANCED RESTRICTION: Force AI models to not turn 180°, limit to 100° at most
        # If car is facing significantly backwards (more than 70 degrees from forward)
        if abs(current_angle) > 70:
            # FORCE SLOW RECOVERY: Gradual angle correction for angles > 70°
            # Use smaller, more controlled turns to gradually reduce the angle
            if current_angle > 70:
                # For angles > 70°, use controlled right turns to slowly recover
                if current_angle > 85:
                    return {'turn_right': True, 'brake': True}  # Strong correction for extreme angles
                else:
                    return {'turn_right': True}  # Gentle correction for moderate angles
            elif current_angle < -70:
                # For angles < -70°, use controlled left turns to slowly recover
                if current_angle < -85:
                    return {'turn_left': True, 'brake': True}  # Strong correction for extreme angles
                else:
                    return {'turn_left': True}  # Gentle correction for moderate angles

        # HARD LIMIT: Prevent 180° turns completely - maximum 100° allowed
        if abs(current_angle) > 100:
            # Emergency correction for extreme angles - force immediate recovery
            if current_angle > 100:
                return {'turn_right': True, 'brake': True}  # Force right turn + brake to reduce angle
            elif current_angle < -100:
                return {'turn_left': True, 'brake': True}   # Force left turn + brake to reduce angle
            else:
                return {'brake': True}  # Emergency brake for extreme cases
        
        rl_action = self.rl_module.get_action(state)
        base_action = self.action_map[rl_action].copy()
        
        # Enhanced safety overrides with swerving preference
        if safety_risks['obstacle_warning']:
            if base_action.get('accelerate', False):
                # Try to swerve instead of just braking
                if safety_risks['swerve_left_possible'] and not safety_risks['near_top_edge']:
                    base_action = {'turn_left': True}
                elif safety_risks['swerve_right_possible'] and not safety_risks['near_bottom_edge']:
                    base_action = {'turn_right': True}
                else:
                    base_action = {'brake': True}
        
        if safety_risks['left_collision'] and base_action.get('change_lane_left', False):
            base_action = {'accelerate': True}
        
        if safety_risks['right_collision'] and base_action.get('change_lane_right', False):
            base_action = {'accelerate': True}
        
        # Prevent excessive turning that could lead to U-turns or road edge hits
        if base_action.get('turn_left', False):
            if current_angle < -45 or safety_risks['near_top_edge']:
                base_action = {'accelerate': True}
        elif base_action.get('turn_right', False):
            if current_angle > 45 or safety_risks['near_bottom_edge']:
                base_action = {'accelerate': True}
        
        # Encourage forward movement when near horizontal boundaries
        if near_horizontal_boundary and not safety_risks['front_collision'] and not safety_risks['road_edge_warning']:
            if not base_action.get('brake', False):
                base_action = {'accelerate': True}
        
        # Use Random Forest for decision refinement if trained
        if self.classifier_trained and not facing_backwards and not safety_risks['road_edge_warning']:
            try:
                features = state.reshape(1, -1)
                decision_probs = self.decision_classifier.predict_proba(features)[0]

                if max(decision_probs) > 0.8:
                    predicted_class = self.decision_classifier.predict(features)[0]
                    if predicted_class == 0 and not near_horizontal_boundary:  # brake
                        base_action = {'brake': True}
                    elif predicted_class == 1:  # accelerate
                        base_action = {'accelerate': True}
                    elif predicted_class == 2 and not safety_risks['left_collision'] and current_angle > -30:
                        base_action = {'turn_left': True}
                    elif predicted_class == 3 and not safety_risks['right_collision'] and current_angle < 30:
                        base_action = {'turn_right': True}
            except Exception as e:
                print(f"Classifier prediction error: {e}")

        if base_action.get('turn_left', False) and current_angle < -45:
            # Don't allow left turn if already facing left (would go more backwards)
            base_action = {'accelerate': True} if car_speed < 3.0 else {'brake': True}

        if base_action.get('turn_right', False) and current_angle > 45:
            # Don't allow right turn if already facing right (would go more backwards)
            base_action = {'accelerate': True} if car_speed < 3.0 else {'brake': True}
        
        return base_action
    
    def predict_speed(self, state: np.ndarray) -> float:
        """Predict optimal speed using Random Forest Regressor"""
        if not self.regressor_trained:
            return 3.0  # default speed
        
        try:
            features = state.reshape(1, -1)
            predicted_speed = self.speed_regressor.predict(features)[0]
            return max(0, min(5, predicted_speed))  # clamp to valid range
        except Exception as e:
            print(f"Speed prediction error: {e}")
            return 3.0
    
    def update_rl(self, state: np.ndarray, action: int, reward: float, 
                  next_state: np.ndarray, done: bool):
        """Update reinforcement learning model"""
        self.rl_module.update_q_table(state, action, reward, next_state, done)
        self.rl_module.remember(state, action, reward, next_state, done)
    
    def collect_training_data(self, state: np.ndarray, action: Dict[str, bool], 
                            speed: float, reward: float):
        """Collect data for training Random Forest models"""
        # Convert action dict to class label
        action_class = 0  # default: brake
        if action.get('accelerate', False):
            action_class = 1
        elif action.get('turn_left', False):
            action_class = 2
        elif action.get('turn_right', False):
            action_class = 3
        elif action.get('change_lane_left', False):
            action_class = 4
        elif action.get('change_lane_right', False):
            action_class = 5
        
        # Store training sample
        sample = {
            'state': state.tolist(),
            'action_class': action_class,
            'speed': speed,
            'reward': reward
        }
        self.training_data.append(sample)
    
    def train_models(self, data: Optional[List[Dict]] = None):
        """Train Random Forest models"""
        if data is None:
            data = self.training_data
        
        if len(data) < 50:  # minimum samples required
            print("Not enough training data. Need at least 50 samples.")
            return False
        
        # Prepare data
        X = np.array([sample['state'] for sample in data])
        y_class = np.array([sample['action_class'] for sample in data])
        y_speed = np.array([sample['speed'] for sample in data])
        
        # Split data
        X_train, X_test, y_class_train, y_class_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )
        _, _, y_speed_train, y_speed_test = train_test_split(
            X, y_speed, test_size=0.2, random_state=42
        )
        
        try:
            # Train classifier
            self.decision_classifier.fit(X_train, y_class_train)
            class_pred = self.decision_classifier.predict(X_test)
            class_accuracy = accuracy_score(y_class_test, class_pred)
            print(f"Decision Classifier Accuracy: {class_accuracy:.3f}")
            self.classifier_trained = True
            
            # Train regressor
            self.speed_regressor.fit(X_train, y_speed_train)
            speed_pred = self.speed_regressor.predict(X_test)
            speed_mse = mean_squared_error(y_speed_test, speed_pred)
            print(f"Speed Regressor MSE: {speed_mse:.3f}")
            self.regressor_trained = True
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def save_models(self, directory: str = "models"):
        """Save trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        try:
            if self.classifier_trained:
                joblib.dump(self.decision_classifier, 
                           os.path.join(directory, "decision_classifier.pkl"))
            
            if self.regressor_trained:
                joblib.dump(self.speed_regressor, 
                           os.path.join(directory, "speed_regressor.pkl"))
            
            # Save Q-table
            joblib.dump(self.rl_module.q_table, 
                       os.path.join(directory, "q_table.pkl"))
            
            print("Models saved successfully!")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, directory: str = "models"):
        """Load pre-trained models"""
        try:
            classifier_path = os.path.join(directory, "decision_classifier.pkl")
            if os.path.exists(classifier_path):
                self.decision_classifier = joblib.load(classifier_path)
                self.classifier_trained = True
                print("Decision classifier loaded!")
            
            regressor_path = os.path.join(directory, "speed_regressor.pkl")
            if os.path.exists(regressor_path):
                self.speed_regressor = joblib.load(regressor_path)
                self.regressor_trained = True
                print("Speed regressor loaded!")
            
            q_table_path = os.path.join(directory, "q_table.pkl")
            if os.path.exists(q_table_path):
                self.rl_module.q_table = joblib.load(q_table_path)
                print("Q-table loaded!")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, any]:
        """Get training statistics"""
        return {
            'total_samples': len(self.training_data),
            'classifier_trained': self.classifier_trained,
            'regressor_trained': self.regressor_trained,
            'q_table_size': len(self.rl_module.q_table),
            'epsilon': self.rl_module.epsilon
        }

class PureMLModule:
    def __init__(self):
        # Random Forest models
        self.decision_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.speed_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # Reinforcement Learning module
        self.rl_module = ReinforcementLearning(state_size=13, action_size=6) 

        self.classifier_trained = False
        self.regressor_trained = False

        self.action_map = {
            0: {'accelerate': True},
            1: {'turn_left': True},
            2: {'turn_right': True},
            3: {'brake': True},
            4: {'change_lane_left': True},
            5: {'change_lane_right': True}
        }

        # Data collection
        self.training_data = []

    def make_decision(self, state: np.ndarray, traffic_light_state: str = "green",
                     distance_to_light: float = 1000) -> Dict[str, bool]:
        # Extract car direction to check for backwards driving
        car_direction_cos = state[3]  
        car_direction_sin = state[4]  
        current_angle = np.arctan2(car_direction_sin, car_direction_cos) * 180 / np.pi

        if abs(current_angle) > 70:
            if current_angle > 70:
                # For angles > 70°, use controlled right turns to slowly recover
                if current_angle > 85:
                    return {'turn_right': True, 'brake': True}  # Strong correction for extreme angles
                else:
                    return {'turn_right': True}  # Gentle correction for moderate angles
            elif current_angle < -70:
                # For angles < -70°, use controlled left turns to slowly recover
                if current_angle < -85:
                    return {'turn_left': True, 'brake': True}  # Strong correction for extreme angles
                else:
                    return {'turn_left': True}  # Gentle correction for moderate angles

        # HARD LIMIT: Prevent 180° turns completely - maximum 100° allowed
        if abs(current_angle) > 100:
            # Emergency correction for extreme angles - force immediate recovery
            if current_angle > 100:
                return {'turn_right': True, 'brake': True}  # Force right turn + brake to reduce angle
            elif current_angle < -100:
                return {'turn_left': True, 'brake': True}   # Force left turn + brake to reduce angle
            else:
                return {'brake': True}  # Emergency brake for extreme cases

        rl_action = self.rl_module.get_action(state)
        base_action = self.action_map[rl_action].copy()

        # Use Random Forest for decision refinement if trained
        if self.classifier_trained:
            try:
                features = state.reshape(1, -1)
                decision_probs = self.decision_classifier.predict_proba(features)[0]

                if max(decision_probs) > 0.7:  
                    predicted_class = self.decision_classifier.predict(features)[0]
                    if predicted_class == 0:  # brake
                        base_action = {'brake': True}
                    elif predicted_class == 1:  # accelerate
                        base_action = {'accelerate': True}
                    elif predicted_class == 2:  # turn left
                        base_action = {'turn_left': True}
                    elif predicted_class == 3:  # turn right
                        base_action = {'turn_right': True}
                    elif predicted_class == 4:  # change lane left
                        base_action = {'change_lane_left': True}
                    elif predicted_class == 5:  # change lane right
                        base_action = {'change_lane_right': True}
            except Exception as e:
                print(f"Classifier prediction error: {e}")

        car_direction_cos = state[3]  
        car_direction_sin = state[4]  
        current_angle = np.arctan2(car_direction_sin, car_direction_cos) * 180 / np.pi

        if base_action.get('turn_left', False) and current_angle < -45:
            base_action = {'accelerate': True}

        if base_action.get('turn_right', False) and current_angle > 45:
            base_action = {'accelerate': True}

        return base_action

    def predict_speed(self, state: np.ndarray) -> float:
        if not self.regressor_trained:
            return 3.0  
        try:
            features = state.reshape(1, -1)
            predicted_speed = self.speed_regressor.predict(features)[0]
            return max(0, min(5, predicted_speed))  
        except Exception as e:
            print(f"Speed prediction error: {e}")
            return 3.0

    def update_rl(self, state: np.ndarray, action: int, reward: float,
                  next_state: np.ndarray, done: bool):
        self.rl_module.update_q_table(state, action, reward, next_state, done)
        self.rl_module.remember(state, action, reward, next_state, done)

    def collect_training_data(self, state: np.ndarray, action: Dict[str, bool],
                             speed: float, reward: float):
        action_class = 0  # default: brake
        if action.get('accelerate', False):
            action_class = 1
        elif action.get('turn_left', False):
            action_class = 2
        elif action.get('turn_right', False):
            action_class = 3
        elif action.get('change_lane_left', False):
            action_class = 4
        elif action.get('change_lane_right', False):
            action_class = 5

        sample = {
            'state': state.tolist(),
            'action_class': action_class,
            'speed': speed,
            'reward': reward
        }
        self.training_data.append(sample)

    def train_models(self, data: Optional[List[Dict]] = None):
        if data is None:
            data = self.training_data

        if len(data) < 50:  # minimum samples required
            print("Not enough training data. Need at least 50 samples.")
            return False

        X = np.array([sample['state'] for sample in data])
        y_class = np.array([sample['action_class'] for sample in data])
        y_speed = np.array([sample['speed'] for sample in data])

        X_train, X_test, y_class_train, y_class_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )
        _, _, y_speed_train, y_speed_test = train_test_split(
            X, y_speed, test_size=0.2, random_state=42
        )

        try:
            # Train classifier
            self.decision_classifier.fit(X_train, y_class_train)
            class_pred = self.decision_classifier.predict(X_test)
            class_accuracy = accuracy_score(y_class_test, class_pred)
            print(f"Decision Classifier Accuracy: {class_accuracy:.3f}")
            self.classifier_trained = True

            # Train regressor
            self.speed_regressor.fit(X_train, y_speed_train)
            speed_pred = self.speed_regressor.predict(X_test)
            speed_mse = mean_squared_error(y_speed_test, speed_pred)
            print(f"Speed Regressor MSE: {speed_mse:.3f}")
            self.regressor_trained = True

            return True

        except Exception as e:
            print(f"Training error: {e}")
            return False

    def save_models(self, directory: str = "models", prefix: str = "pure_ml"):
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            if self.classifier_trained:
                joblib.dump(self.decision_classifier,
                           os.path.join(directory, f"{prefix}_decision_classifier.pkl"))

            if self.regressor_trained:
                joblib.dump(self.speed_regressor,
                           os.path.join(directory, f"{prefix}_speed_regressor.pkl"))

            # Save Q-table
            joblib.dump(self.rl_module.q_table,
                       os.path.join(directory, f"{prefix}_q_table.pkl"))

            print("Pure ML models saved successfully!")
            return True

        except Exception as e:
            print(f"Error saving pure ML models: {e}")
            return False

    def load_models(self, directory: str = "models", prefix: str = "pure_ml"):
        try:
            classifier_path = os.path.join(directory, f"{prefix}_decision_classifier.pkl")
            if os.path.exists(classifier_path):
                self.decision_classifier = joblib.load(classifier_path)
                self.classifier_trained = True
                print("Pure ML decision classifier loaded!")

            regressor_path = os.path.join(directory, f"{prefix}_speed_regressor.pkl")
            if os.path.exists(regressor_path):
                self.speed_regressor = joblib.load(regressor_path)
                self.regressor_trained = True
                print("Pure ML speed regressor loaded!")

            q_table_path = os.path.join(directory, f"{prefix}_q_table.pkl")
            if os.path.exists(q_table_path):
                self.rl_module.q_table = joblib.load(q_table_path)
                print("Pure ML Q-table loaded!")

            return True

        except Exception as e:
            print(f"Error loading pure ML models: {e}")
            return False

    def get_training_stats(self) -> Dict[str, any]:
        return {
            'total_samples': len(self.training_data),
            'classifier_trained': self.classifier_trained,
            'regressor_trained': self.regressor_trained,
            'q_table_size': len(self.rl_module.q_table),
            'epsilon': self.rl_module.epsilon
        }

if __name__ == "__main__":
    # Test the AI module
    ai = AIModule()

    # Test pure ML module
    pure_ml = PureMLModule()
    print("Pure ML module initialized successfully!")