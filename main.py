import pygame
import sys
import numpy as np
import time
from typing import Dict, Optional
import threading


from visuals import Environment, Car
from ai_module import AIModule, PureMLModule
from training import TrainingManager

pygame.init()


BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10
UI_PANEL_WIDTH = 200
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 128, 0)
DARK_RED = (128, 0, 0)

class Button:
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 color: tuple = LIGHT_GRAY, text_color: tuple = BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 24)
        self.clicked = False
        self._text_surface = None
        self._text_rect = None
        self._last_text = None
        self._last_color = None
        self._update_text_surface()
        
    def _update_text_surface(self):
        if (self._last_text != self.text or self._last_color != self.text_color):
            self._text_surface = self.font.render(self.text, True, self.text_color)
            self._text_rect = self._text_surface.get_rect(center=self.rect.center)
            self._last_text = self.text
            self._last_color = self.text_color
        
    def draw(self, screen):
        self._update_text_surface()
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        screen.blit(self._text_surface, self._text_rect)
    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
                return True
        return False

class SimulationGUI:
    
    def __init__(self):
        self.environment = Environment()
        self.ai_module = AIModule()
        self.pure_ml_module = PureMLModule()
        self.training_manager = TrainingManager(self.ai_module)

        self.simulation_running = False
        self.ai_enabled = True
        self.real_time_training = False
        self.manual_control = False
        self.ai_mode = "hybrid"  
        self.mode_switch_counter = 0
        self.mode_switch_interval = 300  

        self.frame_count = 0
        self.start_time = time.time()
        self.last_reward_time = time.time()
        self.total_reward = 0
        self.episode_rewards = []
        
        self.previous_state = None
        self.previous_action_index = 0
        
        self.ui_font = pygame.font.Font(None, 20)
        self.ui_update_counter = 0
        self.cached_stats_surface = None
        self.last_stats_data = None
        
        self.ai_module.load_models()
        self.pure_ml_module.load_models(prefix="pure_ml")
        
        self._create_buttons()
        
        self.screen_width = self.environment.screen.get_width() + UI_PANEL_WIDTH
        self.screen = pygame.display.set_mode((self.screen_width, self.environment.screen.get_height()))
        pygame.display.set_caption("Car Driving AI Simulation - Control Panel")
        
    def _create_buttons(self):
        button_x = self.environment.screen.get_width() + BUTTON_MARGIN
        button_y = BUTTON_MARGIN
        
        self.buttons = {
            'start_stop': Button(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT,
                               "Start Sim", GREEN),
            'ai_mode': Button(button_x, button_y + 50, BUTTON_WIDTH, BUTTON_HEIGHT,
                              "Mode: Hybrid", DARK_GREEN),
            'control_mode': Button(button_x, button_y + 100, BUTTON_WIDTH, BUTTON_HEIGHT,
                                 "Control: AI", DARK_GREEN),
            'train_manual': Button(button_x, button_y + 150, BUTTON_WIDTH, BUTTON_HEIGHT,
                                 "Train Manual", BLUE),
            'train_auto': Button(button_x, button_y + 200, BUTTON_WIDTH, BUTTON_HEIGHT,
                                 "Auto Train: OFF", GRAY),
            'reset_car': Button(button_x, button_y + 250, BUTTON_WIDTH, BUTTON_HEIGHT,
                               "Reset Car", LIGHT_GRAY),
            'save_models': Button(button_x, button_y + 300, BUTTON_WIDTH, BUTTON_HEIGHT,
                                 "Save Models", LIGHT_GRAY),
            'load_models': Button(button_x, button_y + 350, BUTTON_WIDTH, BUTTON_HEIGHT,
                                 "Load Models", LIGHT_GRAY),
            'create_checkpoint': Button(button_x, button_y + 400, BUTTON_WIDTH, BUTTON_HEIGHT,
                                      "Create Checkpoint", LIGHT_GRAY),
            'rollback_data': Button(button_x, button_y + 450, BUTTON_WIDTH, BUTTON_HEIGHT,
                                   "Rollback Data", LIGHT_GRAY),
            'generate_report': Button(button_x, button_y + 500, BUTTON_WIDTH, BUTTON_HEIGHT,
                                    "Generate Report", LIGHT_GRAY)
        }
    
    def handle_button_clicks(self, event):
        for button_name, button in self.buttons.items():
            if button.handle_event(event):
                self._handle_button_action(button_name)
    
    def _handle_button_action(self, button_name: str):
        if button_name == 'start_stop':
            self.simulation_running = not self.simulation_running
            if self.simulation_running:
                self.buttons['start_stop'].text = "Stop Sim"
                self.buttons['start_stop'].color = RED
                self.start_time = time.time()
                self.frame_count = 0
                print("Simulation started")
            else:
                self.buttons['start_stop'].text = "Start Sim"
                self.buttons['start_stop'].color = GREEN
                print("Simulation stopped")
        
        elif button_name == 'ai_mode':
            if self.ai_mode == "hybrid":
                self.ai_mode = "pure_ml"
                self.buttons['ai_mode'].text = "Mode: Pure ML"
                self.buttons['ai_mode'].color = BLUE
            else:
                self.ai_mode = "hybrid"
                self.buttons['ai_mode'].text = "Mode: Hybrid"
                self.buttons['ai_mode'].color = DARK_GREEN
            print(f"AI mode switched to: {self.ai_mode}")
        
        elif button_name == 'control_mode':
            if self.ai_enabled:
                self.ai_enabled = False
                self.manual_control = True
                self.buttons['control_mode'].text = "Control: Manual"
                self.buttons['control_mode'].color = BLUE
            else:
                self.ai_enabled = True
                self.manual_control = False
                self.buttons['control_mode'].text = "Control: AI"
                self.buttons['control_mode'].color = DARK_GREEN
            print(f"Control mode switched to: {'Manual' if self.manual_control else 'AI'}")
        
        elif button_name == 'train_manual':
            print("Starting manual training...")
            success = self.training_manager.manual_train()
            print(f"Manual training {'successful' if success else 'failed'}")
        
        elif button_name == 'train_auto':
            self.real_time_training = not self.real_time_training
            if self.real_time_training:
                self.buttons['train_auto'].text = "Auto Train: ON"
                self.buttons['train_auto'].color = DARK_GREEN
                self.training_manager.start_real_time_training()
                print("Real-time training started")
            else:
                self.buttons['train_auto'].text = "Auto Train: OFF"
                self.buttons['train_auto'].color = GRAY
                self.training_manager.stop_real_time_training()
                print("Real-time training stopped")
        
        elif button_name == 'reset_car':
            self.environment.car.x = 100
            self.environment.car.y = 400
            self.environment.car.angle = 0
            self.environment.car.speed = 0
            self.environment.car.lane = 1
            self.environment.car.target_lane = 1
            print("Car reset to starting position")
        
        elif button_name == 'save_models':
            success = self.ai_module.save_models()
            print(f"Models {'saved successfully' if success else 'save failed'}")
        
        elif button_name == 'load_models':
            success = self.ai_module.load_models()
            print(f"Models {'loaded successfully' if success else 'load failed'}")
        
        elif button_name == 'create_checkpoint':
            success = self.training_manager.create_checkpoint(f"checkpoint_{len(self.training_manager.list_checkpoints())}")
            print(f"Checkpoint {'created' if success else 'failed'}")

        elif button_name == 'rollback_data':
            success = self.training_manager.rollback_to_previous_entries(100)
            print(f"Data rollback {'successful' if success else 'failed'}")

        elif button_name == 'generate_report':
            report = self.training_manager.generate_report()
            print("=== TRAINING REPORT ===")
            print(report)
            with open("training_report.txt", "w") as f:
                f.write(report)
            print("Report saved to training_report.txt")
    
    def handle_manual_control(self, keys):
        """Handle manual keyboard control"""
        if not self.manual_control:
            return
        
        action = {}

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action['accelerate'] = True
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action['brake'] = True
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action['turn_left'] = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action['turn_right'] = True
        if keys[pygame.K_q]:
            action['change_lane_left'] = True
        if keys[pygame.K_e]:
            action['change_lane_right'] = True

        if action:
            self.environment.car.apply_action(action)
    
    def get_traffic_light_info(self) -> tuple:
        car_x, car_y = self.environment.car.x, self.environment.car.y
        
        nearest_light = None
        min_distance = float('inf')
        
        for light in self.environment.traffic_lights:
            distance = np.sqrt((light.x - car_x)**2 + (light.y - car_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_light = light
        
        if nearest_light:
            return nearest_light.state, min_distance
        else:
            return "green", 1000
    
    def calculate_reward(self, action: Dict[str, bool]) -> float:

        reward = 0.0
        car = self.environment.car
        
        # Calculate car direction
        angle_rad = np.radians(car.angle)
        direction_cos = np.cos(angle_rad)
        direction_sin = np.sin(angle_rad)
        
        # Road edge penalties - severe punishment for hitting road boundaries
        if car.y <= 160:  # Too close to top road edge (Y=150)
            reward -= 150  # Massive penalty for hitting top edge
        elif car.y <= 180:  # Warning zone near top edge
            reward -= 50   # Heavy penalty for being too close to top
        elif car.y >= 640:  # Too close to bottom road edge (Y=650)
            reward -= 150  # Massive penalty for hitting bottom edge
        elif car.y >= 620:  # Warning zone near bottom edge
            reward -= 50   # Heavy penalty for being too close to bottom
        

        if 200 <= car.y <= 600:  # Safe zone within road
            reward += 10  # Reward for staying in safe zone
        

        if direction_cos > 0 and 180 <= car.y <= 620:  # Facing forward and on road
            reward += car.speed * 8  # Increased from 4 to 8 - stronger forward movement reward
        elif direction_cos <= 0:  # Facing backwards (U-turn penalty)
            reward -= car.speed * 8  # Heavy penalty for moving backwards
        

        min_sensor = min(car.sensor_readings)
        front_sensors = car.sensor_readings[2:5]  # Front sensors
        min_front_sensor = min(front_sensors)
        

        if min_sensor < 20:
            reward -= 500  # VERY HARSH collision penalty
        elif min_sensor < 40:
            reward -= 300  # VERY HARSH close collision penalty
        elif min_sensor < 60:
            reward -= 150  # VERY HARSH warning penalty
        elif min_sensor < 80:
            reward -= 50   # VERY HARSH early warning penalty
        elif min_sensor > 120:
            reward += 8    # Reward for maintaining very safe distance
        elif min_sensor > 100:
            reward += 5    # Reward for maintaining safe distance
        

        if min_front_sensor < 100:  # Obstacle ahead
            if action.get('turn_left', False) or action.get('turn_right', False):
                # Reward swerving to avoid obstacles
                if min_front_sensor < 80:
                    reward += 25  # High reward for swerving when close
                elif min_front_sensor < 120:
                    reward += 15  # Medium reward for early swerving
                else:
                    reward += 8   # Small reward for preventive swerving
            elif action.get('brake', False):
                # Lower reward for braking (but still positive if necessary)
                if min_front_sensor < 60:
                    reward += 10  # Necessary braking
                elif min_front_sensor < 80:
                    reward += 5   # Early braking (good but less than swerving)
                else:
                    reward -= 2   # Unnecessary braking
        

        if min_front_sensor < 120 and car.speed > 3:  # Fast approach to obstacle
            if action.get('brake', False):
                reward += 12  # Reward early braking when going fast
            elif action.get('turn_left', False) or action.get('turn_right', False):
                reward += 20  # Higher reward for early swerving when going fast
        

        if 180 <= car.y <= 620:
            lane_positions = [200, 400, 600]
            target_y = lane_positions[car.lane]
            lane_deviation = abs(car.y - target_y)
            if lane_deviation < 10:
                reward += 12  # Increased reward for good lane keeping
            elif lane_deviation > 50:
                reward -= 20  # Increased penalty for poor lane keeping
        
        # Traffic light compliance
        light_state, distance_to_light = self.get_traffic_light_info()
        if light_state == "red" and distance_to_light < 100:
            if action.get('brake', False):
                reward += 15  # Good, stopping for red light
            elif action.get('accelerate', False):
                reward -= 50  # Severe penalty for running red light
        
        # U-turn prevention - VERY HARSH penalty for facing backwards
        if abs(car.angle) > 90:
            reward -= 200  # VERY HARSH penalty for U-turns
        elif abs(car.angle) > 45:
            reward -= 100  # VERY HARSH penalty for excessive turning
        
        # Road edge avoidance behavior rewards
        if action.get('turn_left', False) and car.y > 580:  # Turning away from bottom edge
            reward += 20  # Increased reward for avoiding bottom edge
        elif action.get('turn_right', False) and car.y < 220:  # Turning away from top edge
            reward += 20  # Increased reward for avoiding top edge
        elif action.get('turn_left', False) and car.y < 220:  # Turning toward top edge
            reward -= 30  # Penalty for turning toward top edge
        elif action.get('turn_right', False) and car.y > 580:  # Turning toward bottom edge
            reward -= 30  # Penalty for turning toward bottom edge
        
        # Reduced general turning penalties (encourage swerving)
        if action.get('turn_left', False) or action.get('turn_right', False):
            if abs(car.angle) > 30:  # Already turned, don't turn more
                reward -= 4   # Further reduced penalty (was 8)
            elif min_front_sensor > 100:  # Turning without obstacle nearby
                reward -= 0.5  # Very small penalty for unnecessary turning (was 1)
            # No penalty for turning when obstacles are nearby (handled above)
        
        # Horizontal boundary handling (allow looping but discourage edge contact)
        if car.x <= 15 or car.x >= 1185:  # Near horizontal screen edges
            reward -= 15  # Reduced penalty for hitting horizontal borders
        
        # Reward for maintaining reasonable speed and direction within road bounds
        if 2 <= car.speed <= 4 and direction_cos > 0.7 and 180 <= car.y <= 620:
            reward += 30  # Increased from 15 to 30 - much stronger reward for good driving
        elif car.speed > 4.5:
            reward -= 5  # Penalty for excessive speed
        elif car.speed < 1 and not action.get('brake', False):
            reward -= 5  # Penalty for being too slow without reason

        # Additional strong reward for forward acceleration
        if action.get('accelerate', False) and direction_cos > 0.5 and 180 <= car.y <= 620:
            reward += 20  # Strong reward for accelerating forward
        
        # Reward smooth driving (consistent direction) within road bounds
        if abs(car.angle) < 15 and 200 <= car.y <= 600:  # Driving straight in safe zone
            reward += 15  # Increased from 8 to 15 - stronger reward for straight driving
        
        # Penalty for obstacle crashes (additional check) - VERY HARSH
        for obstacle in self.environment.obstacles:
            car_rect = pygame.Rect(car.x - car.width//2, car.y - car.height//2,
                                 car.width, car.height)
            if car_rect.colliderect(obstacle.rect):
                reward -= 1000  # VERY HARSH penalty for actual collision
        
        return reward
    
    def update_ai_control(self):
        """Update AI control system"""
        if not self.ai_enabled or not self.simulation_running:
            return

        ''' Automatic mode switching for balanced data collection - DISABLED

         self.mode_switch_counter += 1
         if self.mode_switch_counter >= self.mode_switch_interval:
             self.mode_switch_counter = 0
             # Switch between hybrid and pure_ml modes
             if self.ai_mode == "hybrid":
                 self.ai_mode = "pure_ml"
                 self.buttons['ai_mode'].text = "Mode: Pure ML"
                 self.buttons['ai_mode'].color = BLUE
             else:
                 self.ai_mode = "hybrid"
                 self.buttons['ai_mode'].text = "Mode: Hybrid"
                 self.buttons['ai_mode'].color = DARK_GREEN'''


        current_state = self.environment.car.get_state_vector()


        light_state, distance_to_light = self.get_traffic_light_info()

        if self.ai_mode == "pure_ml":
            action = self.pure_ml_module.make_decision(current_state, light_state, distance_to_light)
            ai_module = self.pure_ml_module
        else:
            action = self.ai_module.make_decision(current_state, light_state, distance_to_light)
            ai_module = self.ai_module


        self.environment.car.apply_action(action)


        reward = self.calculate_reward(action)
        self.total_reward += reward


        if self.previous_state is not None:

            action_index = 0  
            if action.get('accelerate', False):
                action_index = 0
            elif action.get('turn_left', False):
                action_index = 1
            elif action.get('turn_right', False):
                action_index = 2
            elif action.get('brake', False):
                action_index = 3
            elif action.get('change_lane_left', False):
                action_index = 4
            elif action.get('change_lane_right', False):
                action_index = 5


            collision = min(self.environment.car.sensor_readings) < 30
            traffic_violation = (light_state == "red" and distance_to_light < 100 and
                               action.get('accelerate', False))

            ai_module.update_rl(
                self.previous_state, self.previous_action_index, reward,
                current_state, collision
            )


        if self.simulation_running:
            self.training_manager.log_driving_data(
                current_state, action, self.environment.car.speed, reward,
                light_state, self.ai_mode
            )


        self.previous_state = current_state.copy()
        self.previous_action_index = action_index if 'action_index' in locals() else 0
    
    def draw_ui_panel(self):
        """Draw the UI control panel"""
        panel_x = self.environment.screen.get_width()
        

        pygame.draw.line(self.screen, BLACK, (panel_x, 0),
                        (panel_x, self.screen.get_height()), 2)
        
        for button in self.buttons.values():
            button.draw(self.screen)
        

        self.ui_update_counter += 1
        if self.ui_update_counter >= 5:
            self.ui_update_counter = 0
            self._update_stats_surface(panel_x)
        

        if self.cached_stats_surface:
            self.screen.blit(self.cached_stats_surface, (panel_x + 5, 500))
        else:

            self._update_stats_surface(panel_x)
    
    def _update_stats_surface(self, panel_x):
        """Update the cached stats surface"""
        if self.simulation_running and self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            avg_reward = self.total_reward/self.frame_count
        else:
            elapsed_time = 0
            fps = 0
            avg_reward = 0
        
        # Get current AI stats based on mode
        if self.ai_mode == "pure_ml":
            ai_stats = self.pure_ml_module
        else:
            ai_stats = self.ai_module # Causes lag...

        stats = [
            f"FPS: {fps:.1f}",
            f"Time: {elapsed_time:.1f}s",
            f"Total Reward: {self.total_reward:.1f}",
            f"Avg Reward: {avg_reward:.2f}",
            f"Mode: {self.ai_mode}",
            "",
            "AI Stats:",
            f"Classifier: {'Trained' if ai_stats.classifier_trained else 'Not Trained'}",
            f"Regressor: {'Trained' if ai_stats.regressor_trained else 'Not Trained'}",
            f"Q-States: {len(ai_stats.rl_module.q_table)}",
            f"Epsilon: {ai_stats.rl_module.epsilon:.3f}",
            "",
            "Controls:",
            "WASD/Arrows: Manual",
            "Q/E: Lane Change",
            "Space: Pause",
            "R: Reset Car"
        ]
        

        stats_height = len(stats) * 20
        self.cached_stats_surface = pygame.Surface((UI_PANEL_WIDTH - 10, stats_height))
        self.cached_stats_surface.fill(LIGHT_GRAY)
        
        for i, stat in enumerate(stats):
            color = BLACK if stat else LIGHT_GRAY
            if stat:  
                text = self.ui_font.render(stat, True, color)
                self.cached_stats_surface.blit(text, (0, i * 20))
    
    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        
        print("=== Car Driving AI Simulation ===")
        print("Controls:")
        print("- Click buttons to control simulation")
        print("- WASD/Arrow keys for manual control")
        print("- Space to pause, R to reset car")
        print("- ESC to quit")
        print("=====================================")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.environment.paused = not self.environment.paused
                    elif event.key == pygame.K_r:
                        self._handle_button_action('reset_car')
                
                
                self.handle_button_clicks(event)
            
            
            keys = pygame.key.get_pressed()
            self.handle_manual_control(keys)
            
            
            if self.simulation_running and not self.environment.paused:
                
                self.update_ai_control()
                
                
                self.environment.update()
                
                self.frame_count += 1
            
            
            panel_x = self.environment.screen.get_width()
            pygame.draw.rect(self.screen, LIGHT_GRAY,
                           (panel_x, 0, UI_PANEL_WIDTH, self.screen.get_height()))
            
            
            self.environment.draw()
            
            
            self.screen.blit(self.environment.screen, (0, 0))
            
            
            self.draw_ui_panel()
            
            
            pygame.display.flip()
            clock.tick(60)
        
        print("Shutting down simulation...")
        self.training_manager.cleanup()
        pygame.quit()
        sys.exit()

def main():
    """Main entry point"""
    try:
        sim = SimulationGUI()
        sim.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()