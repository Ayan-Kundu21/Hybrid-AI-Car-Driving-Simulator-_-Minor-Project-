import pygame
import math
import numpy as np
from typing import List, Tuple, Optional


pygame.init()
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

class TrafficLight:
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.state = "green"  # "red", "yellow", "green"
        self.timer = 0
        self.durations = {"green": 300, "yellow": 60, "red": 240}  # frames
        
    def update(self):
        self.timer += 1
        if self.timer >= self.durations[self.state]:
            self.timer = 0
            if self.state == "green":
                self.state = "yellow"
            elif self.state == "yellow":
                self.state = "red"
            else:
                self.state = "green"
    
    def draw(self, screen):
        # Light pole
        pygame.draw.rect(screen, DARK_GRAY, (self.x - 5, self.y, 10, 60))
        
        # Light box
        pygame.draw.rect(screen, BLACK, (self.x - 15, self.y - 45, 30, 45))
        
        # Lights
        colors = [DARK_GRAY, DARK_GRAY, DARK_GRAY]
        if self.state == "red":
            colors[0] = RED
        elif self.state == "yellow":
            colors[1] = YELLOW
        elif self.state == "green":
            colors[2] = GREEN
            
        pygame.draw.circle(screen, colors[0], (int(self.x), int(self.y - 35)), 8)
        pygame.draw.circle(screen, colors[1], (int(self.x), int(self.y - 22)), 8)
        pygame.draw.circle(screen, colors[2], (int(self.x), int(self.y - 9)), 8)

class Obstacle:
    def __init__(self, x: float, y: float, width: float, height: float,
                 speed: float = 1.0, direction: float = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.direction = direction  
        self.rect = pygame.Rect(x, y, width, height)
        
        self.min_x = 50
        self.max_x = SCREEN_WIDTH - 50
        self.min_y = 180  
        self.max_y = 620
        

        self.direction_timer = 0
        self.direction_change_interval = 120  # frames
    
    def update(self):
        angle_rad = math.radians(self.direction)
        self.x += math.cos(angle_rad) * self.speed
        self.y += math.sin(angle_rad) * self.speed
        
        if self.x <= self.min_x or self.x >= self.max_x:
            self.direction = 180 - self.direction
            self.x = max(self.min_x, min(self.max_x, self.x))
        
        if self.y <= self.min_y or self.y >= self.max_y:
            self.direction = -self.direction
            self.y = max(self.min_y, min(self.max_y, self.y))
        
        self.direction_timer += 1
        if self.direction_timer >= self.direction_change_interval:
            self.direction_timer = 0
            import random
            self.direction += random.uniform(-30, 30)
        
        self.rect.x = self.x
        self.rect.y = self.y
    
    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        angle_rad = math.radians(self.direction)
        end_x = self.x + math.cos(angle_rad) * 20
        end_y = self.y + math.sin(angle_rad) * 20
        pygame.draw.line(screen, BLACK, (int(self.x + self.width/2), int(self.y + self.height/2)),
                        (int(end_x + self.width/2), int(end_y + self.height/2)), 2)

class Car:
    
    def __init__(self, x: float, y: float, angle: float = 0):
        self.x = x
        self.y = y
        self.angle = angle  # in degrees
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.deceleration = 0.3
        self.turn_speed = 6  
        
        self.width = 30
        self.height = 15
        
        self.sensor_range = 150
        self.sensor_angles = [-90, -45, -22.5, 0, 22.5, 45, 90]  # relative to car direction
        self.sensor_readings = [self.sensor_range] * len(self.sensor_angles)
        
        self.lane = 1  # 0=left, 1=center, 2=right
        self.target_lane = 1
        
    def update_sensors(self, obstacles: List[Obstacle], traffic_lights: List[TrafficLight], screen_width: int, screen_height: int):
        self.sensor_readings = []
        
        for sensor_angle in self.sensor_angles:
            total_angle = math.radians(self.angle + sensor_angle)
            dx = math.cos(total_angle)
            dy = math.sin(total_angle)
            
            min_distance = self.sensor_range
            
            for step in range(1, int(self.sensor_range)):
                ray_x = self.x + dx * step
                ray_y = self.y + dy * step
                if ray_y <= 0 or ray_y >= screen_height:
                    min_distance = min(min_distance, step)
                    break
                
                wrapped_ray_x = ray_x
                if ray_x < 0:
                    wrapped_ray_x = ray_x + screen_width
                elif ray_x >= screen_width:
                    wrapped_ray_x = ray_x - screen_width
                
                for obstacle in obstacles:
                    if obstacle.rect.collidepoint(wrapped_ray_x, ray_y):
                        min_distance = min(min_distance, step)
                        break
                
                for light in traffic_lights:
                    if light.state == "red":
                        light_rect = pygame.Rect(light.x - 15, light.y - 45, 30, 105)
                        if light_rect.collidepoint(wrapped_ray_x, ray_y):
                            min_distance = min(min_distance, step)
                            break
            
            self.sensor_readings.append(min_distance)
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state as feature vector for AI"""
        state = [
            self.x / SCREEN_WIDTH,  # normalized position
            self.y / SCREEN_HEIGHT,
            self.speed / self.max_speed,  # normalized speed
            math.cos(math.radians(self.angle)),  # direction components
            math.sin(math.radians(self.angle)),
            self.lane / 2,  # normalized lane
        ]
        normalized_sensors = [reading / self.sensor_range for reading in self.sensor_readings]
        state.extend(normalized_sensors)
        
        return np.array(state)
    
    def apply_action(self, action: dict):
        if action.get('accelerate', False):
            self.speed = min(self.max_speed, self.speed + self.acceleration)
        elif action.get('brake', False):
            self.speed = max(0, self.speed - self.deceleration)
        else:
            self.speed = max(0, self.speed - 0.05)
        
        if action.get('turn_left', False):
            self.angle -= self.turn_speed
        elif action.get('turn_right', False):
            self.angle += self.turn_speed
        
        if action.get('change_lane_left', False) and self.lane > 0:
            self.target_lane = self.lane - 1
        elif action.get('change_lane_right', False) and self.lane < 2:
            self.target_lane = self.lane + 1
    
    def update_physics(self):
        prev_x, prev_y = self.x, self.y
        
        angle_rad = math.radians(self.angle)
        self.x += math.cos(angle_rad) * self.speed
        self.y += math.sin(angle_rad) * self.speed
        lane_positions = [200, 400, 600]
        target_y = lane_positions[self.target_lane]
        
        if abs(self.y - target_y) > 5:
            if self.y < target_y:
                self.y += 2
            else:
                self.y -= 2
        else:
            self.lane = self.target_lane
        
        if self.x > SCREEN_WIDTH + self.width//2:
            self.x = -self.width//2
        elif self.x < -self.width//2:
            self.x = SCREEN_WIDTH + self.width//2
            
        self.y = max(180, min(620, self.y))
        
        if self.speed > 0.5:  
            dx = self.x - prev_x
            dy = self.y - prev_y
            
            if abs(dx) > SCREEN_WIDTH / 2:  
                if dx > 0:
                    dx = dx - SCREEN_WIDTH
                else:
                    dx = dx + SCREEN_WIDTH
            
            if abs(dx) > 0.1 or abs(dy) > 0.1: 
                movement_angle = math.degrees(math.atan2(dy, dx))
                
                angle_diff = movement_angle - self.angle
                
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                
                if abs(angle_diff) > 5:  
                    correction_rate = 0.3  
                    self.angle += angle_diff * correction_rate
    
    def draw(self, screen):
        # Car body
        car_points = [
            (-self.width//2, -self.height//2),
            (self.width//2, -self.height//2),
            (self.width//2, self.height//2),
            (-self.width//2, self.height//2)
        ]
        
        # Rotate car points
        angle_rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_points = []
        for px, py in car_points:
            rx = px * cos_a - py * sin_a + self.x
            ry = px * sin_a + py * cos_a + self.y
            rotated_points.append((rx, ry))
        
        pygame.draw.polygon(screen, BLUE, rotated_points)
        pygame.draw.polygon(screen, BLACK, rotated_points, 2)
        
        # Direction indicator
        front_x = self.x + math.cos(angle_rad) * self.width//2
        front_y = self.y + math.sin(angle_rad) * self.width//2
        pygame.draw.circle(screen, WHITE, (int(front_x), int(front_y)), 3)
    
    def draw_sensors(self, screen):
        for i, (sensor_angle, reading) in enumerate(zip(self.sensor_angles, self.sensor_readings)):
            total_angle = math.radians(self.angle + sensor_angle)
            end_x = self.x + math.cos(total_angle) * reading
            end_y = self.y + math.sin(total_angle) * reading
            
            # Color based on distance
            if reading < 50:
                color = RED
            elif reading < 100:
                color = ORANGE
            else:
                color = GREEN
            
            pygame.draw.line(screen, color, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)

class Environment:
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Car Driving AI Simulation")
        self.clock = pygame.time.Clock()
        
        # Initialize objects
        self.car = Car(100, 400, 0)
        self.obstacles = self._create_obstacles()
        self.traffic_lights = self._create_traffic_lights()
        
        # Simulation state
        self.running = True
        self.paused = False
        
    def _create_obstacles(self) -> List[Obstacle]:
        obstacles = []
        import random
        
        # Add moving obstacles with different speeds and directions
        obstacles.append(Obstacle(300, 350, 40, 40, speed=0.8, direction=45))
        obstacles.append(Obstacle(500, 250, 30, 50, speed=1.2, direction=135))
        obstacles.append(Obstacle(700, 450, 50, 30, speed=0.6, direction=270))
        obstacles.append(Obstacle(900, 300, 35, 35, speed=1.0, direction=180))
        obstacles.append(Obstacle(200, 500, 25, 25, speed=1.5, direction=90))
        obstacles.append(Obstacle(600, 200, 45, 35, speed=0.9, direction=315))
        
        return obstacles
    
    def _create_traffic_lights(self) -> List[TrafficLight]:
        lights = []
        lights.append(TrafficLight(400, 150))
        lights.append(TrafficLight(800, 550))
        return lights
    
    def draw_road(self):
        pygame.draw.rect(self.screen, DARK_GRAY, (0, 150, SCREEN_WIDTH, 500))
        
        for y in [275, 425, 575]:
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.rect(self.screen, YELLOW, (x, y, 20, 4))
        
        pygame.draw.rect(self.screen, WHITE, (0, 150, SCREEN_WIDTH, 4))
        pygame.draw.rect(self.screen, WHITE, (0, 646, SCREEN_WIDTH, 4))
    
    def update(self):
        if not self.paused:
            # Update traffic lights
            for light in self.traffic_lights:
                light.update()
            
            # Update moving obstacles
            for obstacle in self.obstacles:
                obstacle.update()
            
            # Update car sensors
            self.car.update_sensors(self.obstacles, self.traffic_lights, SCREEN_WIDTH, SCREEN_HEIGHT)
            
            # Update car physics
            self.car.update_physics()
    
    def draw(self):
        self.screen.fill(WHITE)
        
        self.draw_road()
        
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        for light in self.traffic_lights:
            light.draw(self.screen)
        
        self.car.draw_sensors(self.screen)
        self.car.draw(self.screen)
        
        self._draw_ui()
        
        pygame.display.flip()
    
    def _draw_ui(self):
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 20)

        # HUD Background
        hud_bg = pygame.Surface((250, 220))
        hud_bg.fill((240, 240, 240))
        hud_bg.set_alpha(200)
        pygame.draw.rect(hud_bg, BLACK, (0, 0, 250, 220), 2)
        self.screen.blit(hud_bg, (10, 10))

        # HUD Title
        hud_title = font.render("CAR DRIVING AI HUD", True, BLACK)
        self.screen.blit(hud_title, (20, 20))

        # Car info
        speed_text = font.render(f"Speed: {self.car.speed:.1f}", True, BLACK)
        lane_text = font.render(f"Lane: {self.car.lane}", True, BLACK)

        self.screen.blit(speed_text, (20, 50))
        self.screen.blit(lane_text, (20, 80))

        # Sensor readings with enhanced display
        sensor_text = font.render("Sensor Distances:", True, BLACK)
        self.screen.blit(sensor_text, (20, 110))

        # Draw sensor bars
        sensor_y = 140
        for i, reading in enumerate(self.car.sensor_readings):
            # Sensor label
            color = RED if reading < 50 else ORANGE if reading < 100 else GREEN
            label = small_font.render(f"S{i}:", True, BLACK)
            self.screen.blit(label, (20, sensor_y))

            # Sensor value
            value_text = small_font.render(f"{reading:.0f}", True, color)
            self.screen.blit(value_text, (60, sensor_y))

            # Visual bar
            bar_width = min(150, reading * 1.5)
            pygame.draw.rect(self.screen, color, (100, sensor_y + 5, bar_width, 10))
            pygame.draw.rect(self.screen, BLACK, (100, sensor_y + 5, 150, 10), 1)

            sensor_y += 25

        # Traffic light indicator
        light_x, light_y = 20, sensor_y + 10
        light_text = small_font.render("Traffic Light:", True, BLACK)
        self.screen.blit(light_text, (light_x, light_y))

        # Find nearest traffic light
        car_x, car_y = self.car.x, self.car.y
        nearest_light = None
        min_distance = float('inf')

        for light in self.traffic_lights:
            distance = np.sqrt((light.x - car_x)**2 + (light.y - car_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_light = light

        if nearest_light:
            light_state_text = small_font.render(f"State: {nearest_light.state}", True, BLACK)
            light_dist_text = small_font.render(f"Dist: {min_distance:.0f}", True, BLACK)
            self.screen.blit(light_state_text, (light_x, light_y + 25))
            self.screen.blit(light_dist_text, (light_x, light_y + 50))

            # Draw light indicator
            light_color = GREEN if nearest_light.state == "green" else YELLOW if nearest_light.state == "yellow" else RED
            pygame.draw.circle(self.screen, light_color, (light_x + 120, light_y + 35), 8)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    # Reset car position
                    self.car.x = 100
                    self.car.y = 400
                    self.car.angle = 0
                    self.car.speed = 0
    
    def run_frame(self):
        self.handle_events()
        self.update()
        self.draw()
        self.clock.tick(FPS)
        return self.running

if __name__ == "__main__":
    env = Environment()
    while env.run_frame():
        pass
    pygame.quit()