

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from ai_module import AIModule, PureMLModule
from training import TrainingManager
import json
import time

def evaluate_ai_modes():


    print("=== AI Mode Performance Evaluation ===")
    print("Loading AI modules...")


    hybrid_ai = AIModule()
    pure_ml_ai = PureMLModule()

    print("OK Both AI modules loaded successfully")

    try:
        test_data = pd.read_csv('logs/comprehensive_training_data.csv')
        print(f"OK Loaded {len(test_data)} test samples")

        feature_columns = ['car_position_x', 'car_position_y', 'speed', 'angle', 'lane'] + \
                         [f'sensor_{i}' for i in range(7)]

   
        test_data['sensor_distances_parsed'] = test_data['sensor_distances'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )


        test_features = []
        test_decision_labels = []
        test_speed_labels = []

        for _, row in test_data.iterrows():

            basic_features = [
                row['car_position_x'],
                row['car_position_y'],
                row['speed'],
                np.radians(row['angle']), 
                row.get('lane', 0.5)  
            ]


            sensor_features = row['sensor_distances_parsed']
            full_features = basic_features + sensor_features

            test_features.append(full_features)

            action_str = row['action_taken'].replace('true', 'True').replace('false', 'False')
            action = eval(action_str)
            if 'accelerate' in action and action['accelerate']:
                decision_label = 0  # accelerate
            elif 'brake' in action and action['brake']:
                decision_label = 1  # brake
            elif 'turn_left' in action and action['turn_left']:
                decision_label = 2  # turn_left
            elif 'turn_right' in action and action['turn_right']:
                decision_label = 3  # turn_right
            elif 'change_lane_left' in action and action['change_lane_left']:
                decision_label = 4  # change_lane_left
            elif 'change_lane_right' in action and action['change_lane_right']:
                decision_label = 5  # change_lane_right
            else:
                decision_label = 0  # default to accelerate

            test_decision_labels.append(decision_label)
            test_speed_labels.append(row['speed'])

        test_features = np.array(test_features)
        test_decision_labels = np.array(test_decision_labels)
        test_speed_labels = np.array(test_speed_labels)

        print(f" Prepared {len(test_features)} test samples with {test_features.shape[1]} features")

    except Exception as e:
        print(f" Error loading test data: {e}")
        print("Creating synthetic test data...")
        test_features, test_decision_labels, test_speed_labels = create_synthetic_test_data()
        print(f" Created {len(test_features)} synthetic test samples")

    # Evaluate Hybrid AI
    print("\n--- Evaluating Hybrid AI ---")
    hybrid_start_time = time.time()

    hybrid_decision_preds = hybrid_ai.decision_classifier.predict(test_features)
    hybrid_speed_preds = hybrid_ai.speed_regressor.predict(test_features)

    hybrid_decision_acc = accuracy_score(test_decision_labels, hybrid_decision_preds)
    hybrid_speed_mse = mean_squared_error(test_speed_labels, hybrid_speed_preds)
    hybrid_inference_time = time.time() - hybrid_start_time

    print(f"Decision Accuracy: {hybrid_decision_acc:.3f}")
    print(f"Speed Prediction MSE: {hybrid_speed_mse:.3f}")
    print(f"Inference Time: {hybrid_inference_time:.3f}s")

    # Evaluate Pure ML
    print("\n--- Evaluating Pure ML ---")
    pure_ml_start_time = time.time()

    pure_ml_decision_preds = pure_ml_ai.decision_classifier.predict(test_features)
    pure_ml_speed_preds = pure_ml_ai.speed_regressor.predict(test_features)

    pure_ml_decision_acc = accuracy_score(test_decision_labels, pure_ml_decision_preds)
    pure_ml_speed_mse = mean_squared_error(test_speed_labels, pure_ml_speed_preds)
    pure_ml_inference_time = time.time() - pure_ml_start_time

    print(f"Decision Accuracy: {pure_ml_decision_acc:.3f}")
    print(f"Speed Prediction MSE: {pure_ml_speed_mse:.3f}")
    print(f"Inference Time: {pure_ml_inference_time:.3f}s")

    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"Decision Accuracy Improvement: {((pure_ml_decision_acc - hybrid_decision_acc) / hybrid_decision_acc * 100):.1f}%")
    print(f"Speed Prediction Improvement: {((hybrid_speed_mse - pure_ml_speed_mse) / hybrid_speed_mse * 100):.1f}%")
    print(f"Inference Speed Ratio: {hybrid_inference_time / pure_ml_inference_time:.2f}x")

    print("\n--- Scenario-Based Testing ---")
    test_scenarios = [
        ("Normal driving", [0.5, 0.5, 2.0, 0.0, 0.5] + [1.0]*7),
        ("Collision avoidance", [0.3, 0.3, 1.0, 0.0, 0.5] + [0.1, 0.1, 0.1, 0.3, 0.5, 0.7, 0.9]),
        ("Traffic light stop", [0.7, 0.5, 0.5, 0.0, 0.5] + [0.8]*7),
        ("High speed", [0.5, 0.5, 4.5, 0.0, 0.5] + [1.0]*7),
        ("Sensor failure", [0.5, 0.5, 1.0, 0.0, 0.5] + [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    ]

    for scenario_name, scenario_features in test_scenarios:
        print(f"\n{scenario_name}:")
        hybrid_decision = hybrid_ai.decision_classifier.predict([scenario_features])[0]
        pure_ml_decision = pure_ml_ai.decision_classifier.predict([scenario_features])[0]

        hybrid_speed = hybrid_ai.speed_regressor.predict([scenario_features])[0]
        pure_ml_speed = pure_ml_ai.speed_regressor.predict([scenario_features])[0]

        decision_map = {0: 'accelerate', 1: 'brake', 2: 'turn_left', 3: 'turn_right', 4: 'change_lane_left', 5: 'change_lane_right'}

        print(f"  Hybrid: {decision_map.get(hybrid_decision, 'unknown')} @ {hybrid_speed:.2f} speed")
        print(f"  Pure ML: {decision_map.get(pure_ml_decision, 'unknown')} @ {pure_ml_speed:.2f} speed")

        if hybrid_decision == pure_ml_decision:
            print("   Both modes agree on decision")
        else:
            print("   Modes disagree on decision")

    # Saveresults
    evaluation_results = {
        'timestamp': time.time(),
        'hybrid_ai': {
            'decision_accuracy': hybrid_decision_acc,
            'speed_mse': hybrid_speed_mse,
            'inference_time': hybrid_inference_time
        },
        'pure_ml_ai': {
            'decision_accuracy': pure_ml_decision_acc,
            'speed_mse': pure_ml_speed_mse,
            'inference_time': pure_ml_inference_time
        },
        'test_samples': len(test_features),
        'features_per_sample': test_features.shape[1]
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n Evaluation results saved to evaluation_results.json")

    print("\n=== Final Recommendation ===")
    if pure_ml_decision_acc > hybrid_decision_acc and pure_ml_speed_mse < hybrid_speed_mse:
        print(" Pure ML mode shows better overall performance")
        print(" Recommend using Pure ML for production")
    elif hybrid_decision_acc > pure_ml_decision_acc and hybrid_speed_mse < pure_ml_speed_mse:
        print(" Hybrid mode shows better overall performance")
        print(" Recommend using Hybrid for production")
    else:
        print(" Both modes have trade-offs:")
        print(f"   Hybrid: Better decision accuracy ({hybrid_decision_acc:.3f} vs {pure_ml_decision_acc:.3f})")
        print(f"   Pure ML: Better speed prediction ({pure_ml_speed_mse:.3f} vs {hybrid_speed_mse:.3f})")

    print("\n Evaluation complete!")

def create_synthetic_test_data(num_samples=1000):
    test_features = []
    test_decision_labels = []
    test_speed_labels = []

    for _ in range(num_samples):
        x_pos = np.random.uniform(0.1, 0.9)
        y_pos = np.random.uniform(0.2, 0.8)
        speed = np.random.uniform(0.5, 5.0)
        angle = np.random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
        lane = np.random.choice([0.0, 0.5, 1.0])

        sensors = [np.random.uniform(0.1, 1.0) for _ in range(7)]

        features = [x_pos, y_pos, speed, angle, lane] + sensors
        test_features.append(features)

        decision = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        test_decision_labels.append(decision)
        test_speed_labels.append(speed)

    return np.array(test_features), np.array(test_decision_labels), np.array(test_speed_labels)

if __name__ == "__main__":
    evaluate_ai_modes()