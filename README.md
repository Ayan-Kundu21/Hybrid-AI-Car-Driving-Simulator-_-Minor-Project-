Car Driving AI Simulation

A comprehensive 2D car driving simulation with intelligent AI systems using Random Forest models, reinforcement learning, and rule-based safety modules.

Features:

Core AI Components
- Hybrid AI System: Combines rule-based safety with ML predictions
- Pure ML System: ML-only decision making for performance comparison
- Rule-based Safety Module: Lane keeping, traffic signal compliance, collision avoidance
- Random Forest Classifier: Decision making for turning, stopping, overtaking
- Random Forest Regressor: Speed prediction and control
- Reinforcement Learning: Q-learning for adaptive behavior and lane changes
- Real-time Training: Automated model retraining from collected driving data

Simulation Environment
- 2D Pygame Visualization: Roads, traffic lights, moving obstacles, and car rendering
- Ray-casting Sensors: 7-directional obstacle detection with 150-pixel range
- Dynamic Traffic Lights: Realistic timing cycles (green/yellow/red)
- Moving Obstacles: Dynamic obstacles with collision avoidance for enhanced training
- Infinite Path: Car loops seamlessly at screen boundaries for continuous driving
- Interactive GUI: Optimized, flicker-free control panel with all simulation functions

Training Systems
- Manual Training: Train models from collected data on demand
- Automated Training: Background training every 5 minutes with sufficient new data
- Data Logging: JSON and CSV storage of driving states, actions, and rewards
- Performance Evaluation: Model accuracy and prediction error metrics
- Data Rollback: Revert to previous checkpoints or specific entry counts
- Visualization Module: Comprehensive statistical analysis and plotting

Installation

1. Clone or download the project files
2. Install dependencies:
   bash>
   pip install -r requirements.txt

Usage:

Running the Simulation
bash>
python main.py


Controls:

GUI Buttons
- Start/Stop Sim: Toggle simulation running state
- Mode: Hybrid/Pure ML: Switch between hybrid AI and pure ML modes
- Manual: ON/OFF: Enable manual keyboard control
- Train Manual: Trigger manual training from collected data
- Auto Train: ON/OFF: Toggle automated real-time training
- Reset Car: Reset car to starting position
- Save Models: Save trained AI models to disk
- Load Models: Load pre-trained models
- Create Checkpoint: Create data checkpoint for rollback
- Rollback Data: Revert to previous data state
- Generate Report: Create comprehensive training report

Keyboard Controls (Manual Mode)
- WASD: Accelerate, brake, steer
- Q/E: Change lanes left/right
- Space: Pause/unpause simulation
- R: Reset car position
- ESC: Exit simulation

File Structure


Car_Driving_AI_Simulation/
|── main.py             
|── visuals.py           
|── ai_module.py         
|── training.py          
|── data.py              
|── requirements.txt     
|── README.md           
|── logs/               
│   |── driving_data.json
│   L── driving_data.csv
|── models/             
│   |── decision_classifier.pkl
│   |── speed_regressor.pkl
│   |── q_table.pkl
│   |── pure_ml_decision_classifier.pkl
│   |── pure_ml_speed_regressor.pkl
│   L── pure_ml_q_table.pkl
|── training_report.txt 
L── visualizations/     


Training Data Format:

The system logs driving data in both JSON and CSV formats with the following structure:

JSON Format
json>
{
  "timestamp": 1756111812.5800483,
  "state": [0.083, 0.5, 0.0, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "action": {"accelerate": true},
  "speed": 0.2,
  "reward": 5.4,
  "traffic_light_state": "green",
  "ai_mode": "hybrid"
}


CSV Format
The CSV format includes flattened columns for easier analysis:
- `timestamp`: Unix timestamp
- `car_position_x`, `car_position_y`: Normalized coordinates
- `angle`: Car orientation in degrees
- `speed`: Current speed
- `sensor_distances`: JSON array of sensor readings
- `traffic_light_state`: Current traffic light state
- `action_taken`: JSON object of actions
- `reward`: Reward value
- `ai_mode`: "hybrid" or "pure_ml"

Data Visualization:

The `data.py` module provides comprehensive visualization capabilities:

python>
from data import DataVisualizer

Initialize and load data
Generate visualizations
Show interactive dashboard
Generate statistical report


Data Rollback Features:

The training system now supports data rollback:
>python
Create checkpoint
Rollback to checkpoint
Rollback to specific entry count
List available checkpoints

AI Mode Comparison:

The system now supports switching between hybrid and pure ML modes:

python>

Switch to pure ML mode (no safety overrides)
Switch back to hybrid mode (with safety overrides)

HUD Display

The visuals module includes a HUD with:
- Real-time sensor visualizations
- Traffic light indicators
- Performance metrics
- Visual sensor distance bars

Performance Metrics

The system tracks and visualizes:
- Accuracy over time
- Reward improvement
- Failure and struggle counts
- Task completion time
- Safety violations
- Performance comparison between hybrid and pure ML models

Requirements:

- Python 3.8+
- Pygame 2.5.2
- scikit-learn 1.3.2
- numpy 1.24.3
- pandas 2.0.3
- matplotlib 3.7.2
- joblib 1.3.2
