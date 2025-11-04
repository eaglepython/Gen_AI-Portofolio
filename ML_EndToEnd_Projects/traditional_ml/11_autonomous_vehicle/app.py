"""
Autonomous Vehicle AI System - Complete Implementation
Advanced autonomous vehicle system with sensor fusion, path planning, and decision making.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import xgboost as xgb

# Computer vision
import cv2
try:
    from ultralytics import YOLO
except:
    logger.warning("YOLO not available")

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Input, concatenate
from tensorflow.keras.optimizers import Adam

# Path planning
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
import networkx as nx

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Tuple
import uvicorn

# Utilities
import joblib
import pickle
import json
from pathlib import Path
import logging
import asyncio
import random
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorDataGenerator:
    """Generate realistic autonomous vehicle sensor data."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_lidar_data(self, n_points: int = 1000, scan_range: float = 100.0) -> Dict:
        """Generate LIDAR point cloud data."""
        # Simulate 360-degree LIDAR scan
        angles = np.linspace(0, 2*np.pi, n_points)
        
        # Generate distances with obstacles
        distances = []
        for angle in angles:
            # Base distance (road boundaries)
            base_distance = scan_range * 0.8
            
            # Add obstacles (cars, buildings, etc.)
            obstacle_prob = 0.15
            if np.random.random() < obstacle_prob:
                obstacle_distance = np.random.uniform(5, 50)
                distances.append(min(obstacle_distance, base_distance))
            else:
                # Add noise to base distance
                noise = np.random.normal(0, 2)
                distances.append(min(scan_range, max(1, base_distance + noise)))
        
        # Convert to Cartesian coordinates
        x_coords = np.array(distances) * np.cos(angles)
        y_coords = np.array(distances) * np.sin(angles)
        
        return {
            'angles': angles.tolist(),
            'distances': distances,
            'x_coords': x_coords.tolist(),
            'y_coords': y_coords.tolist(),
            'point_count': n_points,
            'max_range': scan_range
        }
    
    def generate_camera_data(self, width: int = 640, height: int = 480) -> Dict:
        """Generate camera sensor data."""
        # Simulate detected objects
        objects = []
        object_types = ['car', 'truck', 'pedestrian', 'bicycle', 'traffic_sign', 'traffic_light']
        
        n_objects = np.random.randint(0, 8)
        
        for _ in range(n_objects):
            obj_type = np.random.choice(object_types)
            
            # Object bounding box
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(30, 200)
            h = np.random.randint(30, 150)
            
            # Distance estimation
            distance = np.random.uniform(5, 100)
            
            # Confidence
            confidence = np.random.uniform(0.6, 0.99)
            
            objects.append({
                'type': obj_type,
                'bbox': [x, y, w, h],
                'distance': round(distance, 2),
                'confidence': round(confidence, 3),
                'relative_velocity': round(np.random.normal(0, 5), 2)  # m/s
            })
        
        # Lane detection
        lane_lines = []
        for i in range(2, 4):  # 2-3 lane lines typically visible
            # Polynomial coefficients for lane line
            coeffs = [
                np.random.normal(0, 0.001),  # Curvature
                np.random.normal(0, 0.1),    # Slope
                np.random.uniform(-200, 200)  # Intercept
            ]
            
            lane_lines.append({
                'coefficients': coeffs,
                'confidence': round(np.random.uniform(0.7, 0.95), 3)
            })
        
        return {
            'resolution': [width, height],
            'detected_objects': objects,
            'lane_lines': lane_lines,
            'brightness': round(np.random.uniform(0.3, 1.0), 2),
            'weather_condition': np.random.choice(['clear', 'cloudy', 'rain', 'fog'])
        }
    
    def generate_radar_data(self, max_range: float = 200.0) -> Dict:
        """Generate radar sensor data."""
        # Radar detects objects with velocity
        detections = []
        n_detections = np.random.randint(0, 15)
        
        for _ in range(n_detections):
            # Polar coordinates
            range_val = np.random.uniform(5, max_range)
            azimuth = np.random.uniform(-60, 60)  # degrees
            
            # Velocity (Doppler)
            radial_velocity = np.random.normal(0, 10)  # m/s
            
            # RCS (Radar Cross Section)
            rcs = np.random.uniform(-30, 20)  # dBsm
            
            detections.append({
                'range': round(range_val, 2),
                'azimuth': round(azimuth, 2),
                'radial_velocity': round(radial_velocity, 2),
                'rcs': round(rcs, 2),
                'snr': round(np.random.uniform(10, 40), 2)  # Signal-to-noise ratio
            })
        
        return {
            'detections': detections,
            'max_range': max_range,
            'resolution': {
                'range': 0.1,  # meters
                'azimuth': 0.5  # degrees
            }
        }
    
    def generate_imu_data(self) -> Dict:
        """Generate IMU (Inertial Measurement Unit) data."""
        return {
            'acceleration': {
                'x': round(np.random.normal(0, 0.5), 3),  # m/s²
                'y': round(np.random.normal(0, 0.3), 3),
                'z': round(np.random.normal(9.81, 0.2), 3)  # Include gravity
            },
            'angular_velocity': {
                'x': round(np.random.normal(0, 0.1), 3),  # rad/s
                'y': round(np.random.normal(0, 0.1), 3),
                'z': round(np.random.normal(0, 0.2), 3)
            },
            'orientation': {
                'roll': round(np.random.normal(0, 2), 2),    # degrees
                'pitch': round(np.random.normal(0, 1), 2),
                'yaw': round(np.random.uniform(0, 360), 2)
            }
        }
    
    def generate_gps_data(self, base_lat: float = 37.7749, base_lon: float = -122.4194) -> Dict:
        """Generate GPS data."""
        # Add small random variations to base coordinates
        lat_offset = np.random.normal(0, 0.001)  # ~100m variation
        lon_offset = np.random.normal(0, 0.001)
        
        return {
            'latitude': round(base_lat + lat_offset, 6),
            'longitude': round(base_lon + lon_offset, 6),
            'altitude': round(np.random.normal(50, 10), 2),  # meters
            'accuracy': round(np.random.uniform(1, 5), 2),   # meters
            'speed': round(np.random.uniform(0, 30), 2),     # m/s
            'heading': round(np.random.uniform(0, 360), 2),  # degrees
            'satellite_count': np.random.randint(8, 15)
        }
    
    def generate_vehicle_state(self) -> Dict:
        """Generate vehicle state data."""
        return {
            'speed': round(np.random.uniform(0, 30), 2),        # m/s
            'steering_angle': round(np.random.normal(0, 10), 2), # degrees
            'throttle': round(np.random.uniform(0, 1), 3),      # 0-1
            'brake': round(np.random.uniform(0, 0.2), 3),       # 0-1
            'gear': np.random.choice(['P', 'R', 'N', 'D']),
            'fuel_level': round(np.random.uniform(0.2, 1), 2),  # 0-1
            'engine_rpm': round(np.random.uniform(800, 3000)),
            'battery_voltage': round(np.random.uniform(12, 14), 2),
            'tire_pressure': {
                'front_left': round(np.random.uniform(30, 35), 1),
                'front_right': round(np.random.uniform(30, 35), 1),
                'rear_left': round(np.random.uniform(30, 35), 1),
                'rear_right': round(np.random.uniform(30, 35), 1)
            }
        }
    
    def generate_complete_sensor_frame(self, timestamp: datetime = None) -> Dict:
        """Generate a complete sensor data frame."""
        if timestamp is None:
            timestamp = datetime.now()
        
        return {
            'timestamp': timestamp.isoformat(),
            'lidar': self.generate_lidar_data(),
            'camera': self.generate_camera_data(),
            'radar': self.generate_radar_data(),
            'imu': self.generate_imu_data(),
            'gps': self.generate_gps_data(),
            'vehicle_state': self.generate_vehicle_state()
        }


class SensorFusionEngine:
    """Multi-sensor data fusion for autonomous vehicles."""
    
    def __init__(self):
        self.kalman_filters = {}
        self.object_tracks = {}
        self.track_id_counter = 0
        
    def fuse_sensor_data(self, sensor_data: Dict) -> Dict:
        """Fuse data from multiple sensors."""
        # Extract data from each sensor
        lidar_data = sensor_data.get('lidar', {})
        camera_data = sensor_data.get('camera', {})
        radar_data = sensor_data.get('radar', {})
        
        # Object detection fusion
        fused_objects = self.fuse_object_detections(camera_data, radar_data, lidar_data)
        
        # Localization fusion
        fused_pose = self.fuse_localization_data(sensor_data.get('gps', {}), 
                                                sensor_data.get('imu', {}))
        
        # Environmental understanding
        environment_state = self.analyze_environment(sensor_data)
        
        return {
            'fused_objects': fused_objects,
            'vehicle_pose': fused_pose,
            'environment': environment_state,
            'sensor_health': self.check_sensor_health(sensor_data),
            'confidence': self.calculate_fusion_confidence(sensor_data)
        }
    
    def fuse_object_detections(self, camera_data: Dict, radar_data: Dict, lidar_data: Dict) -> List[Dict]:
        """Fuse object detections from multiple sensors."""
        fused_objects = []
        
        # Start with camera detections
        camera_objects = camera_data.get('detected_objects', [])
        radar_detections = radar_data.get('detections', [])
        
        for cam_obj in camera_objects:
            # Create fused object starting with camera data
            fused_obj = {
                'id': self.get_next_track_id(),
                'type': cam_obj['type'],
                'distance': cam_obj['distance'],
                'confidence': cam_obj['confidence'],
                'velocity': cam_obj.get('relative_velocity', 0),
                'bbox': cam_obj['bbox'],
                'sensors_detected': ['camera']
            }
            
            # Try to match with radar detections
            for radar_det in radar_detections:
                radar_distance = radar_det['range']
                radar_angle = radar_det['azimuth']
                
                # Simple distance-based matching (in practice, would use more sophisticated methods)
                if abs(radar_distance - cam_obj['distance']) < 10:
                    fused_obj['distance'] = (radar_distance + cam_obj['distance']) / 2
                    fused_obj['velocity'] = radar_det['radial_velocity']
                    fused_obj['confidence'] = min(0.99, fused_obj['confidence'] + 0.1)
                    fused_obj['sensors_detected'].append('radar')
                    break
            
            # Add LIDAR validation if available
            if lidar_data.get('distances'):
                # Simplified LIDAR validation
                fused_obj['sensors_detected'].append('lidar')
                fused_obj['confidence'] = min(0.99, fused_obj['confidence'] + 0.05)
            
            fused_objects.append(fused_obj)
        
        return fused_objects
    
    def fuse_localization_data(self, gps_data: Dict, imu_data: Dict) -> Dict:
        """Fuse GPS and IMU data for vehicle localization."""
        # Simple fusion (in practice, would use Extended Kalman Filter)
        pose = {
            'latitude': gps_data.get('latitude', 0),
            'longitude': gps_data.get('longitude', 0),
            'altitude': gps_data.get('altitude', 0),
            'heading': gps_data.get('heading', 0),
            'speed': gps_data.get('speed', 0),
            'accuracy': gps_data.get('accuracy', 10)
        }
        
        # Incorporate IMU data
        if imu_data:
            # Use IMU for short-term accuracy
            pose['roll'] = imu_data.get('orientation', {}).get('roll', 0)
            pose['pitch'] = imu_data.get('orientation', {}).get('pitch', 0)
            
            # IMU can provide more accurate heading in short term
            imu_yaw = imu_data.get('orientation', {}).get('yaw', 0)
            if abs(imu_yaw - pose['heading']) < 10:  # If readings are close
                pose['heading'] = (pose['heading'] + imu_yaw) / 2
            
            pose['accuracy'] = min(pose['accuracy'], 2.0)  # IMU improves accuracy
        
        return pose
    
    def analyze_environment(self, sensor_data: Dict) -> Dict:
        """Analyze environmental conditions."""
        camera_data = sensor_data.get('camera', {})
        vehicle_state = sensor_data.get('vehicle_state', {})
        
        # Weather analysis
        weather = camera_data.get('weather_condition', 'clear')
        brightness = camera_data.get('brightness', 1.0)
        
        # Road analysis
        lane_lines = camera_data.get('lane_lines', [])
        lane_confidence = np.mean([line['confidence'] for line in lane_lines]) if lane_lines else 0.5
        
        # Traffic analysis
        objects = camera_data.get('detected_objects', [])
        traffic_density = len([obj for obj in objects if obj['type'] in ['car', 'truck']]) / 10.0
        
        return {
            'weather_condition': weather,
            'visibility': brightness,
            'lane_detection_confidence': round(lane_confidence, 3),
            'traffic_density': min(1.0, traffic_density),
            'road_type': 'highway' if vehicle_state.get('speed', 0) > 20 else 'city',
            'driving_conditions': self.assess_driving_conditions(weather, brightness, traffic_density)
        }
    
    def assess_driving_conditions(self, weather: str, brightness: float, traffic_density: float) -> str:
        """Assess overall driving conditions."""
        score = 1.0
        
        if weather in ['rain', 'fog']:
            score -= 0.3
        elif weather == 'cloudy':
            score -= 0.1
        
        if brightness < 0.5:
            score -= 0.2
        
        if traffic_density > 0.7:
            score -= 0.2
        
        if score > 0.8:
            return 'excellent'
        elif score > 0.6:
            return 'good'
        elif score > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def check_sensor_health(self, sensor_data: Dict) -> Dict:
        """Check health status of all sensors."""
        health = {}
        
        # LIDAR health
        lidar_data = sensor_data.get('lidar', {})
        lidar_points = lidar_data.get('point_count', 0)
        health['lidar'] = 'healthy' if lidar_points > 800 else 'degraded'
        
        # Camera health
        camera_data = sensor_data.get('camera', {})
        camera_objects = len(camera_data.get('detected_objects', []))
        health['camera'] = 'healthy' if camera_objects >= 0 else 'offline'
        
        # Radar health
        radar_data = sensor_data.get('radar', {})
        radar_detections = len(radar_data.get('detections', []))
        health['radar'] = 'healthy' if radar_detections >= 0 else 'offline'
        
        # GPS health
        gps_data = sensor_data.get('gps', {})
        gps_accuracy = gps_data.get('accuracy', 100)
        health['gps'] = 'healthy' if gps_accuracy < 5 else 'degraded'
        
        # IMU health
        imu_data = sensor_data.get('imu', {})
        health['imu'] = 'healthy' if imu_data else 'offline'
        
        return health
    
    def calculate_fusion_confidence(self, sensor_data: Dict) -> float:
        """Calculate overall sensor fusion confidence."""
        sensor_health = self.check_sensor_health(sensor_data)
        
        healthy_sensors = sum(1 for status in sensor_health.values() if status == 'healthy')
        total_sensors = len(sensor_health)
        
        base_confidence = healthy_sensors / total_sensors
        
        # Adjust based on environmental conditions
        environment = self.analyze_environment(sensor_data)
        condition_factor = {
            'excellent': 1.0,
            'good': 0.9,
            'fair': 0.8,
            'poor': 0.6
        }
        
        driving_condition = environment.get('driving_conditions', 'fair')
        confidence = base_confidence * condition_factor.get(driving_condition, 0.8)
        
        return round(confidence, 3)
    
    def get_next_track_id(self) -> int:
        """Get next available track ID."""
        self.track_id_counter += 1
        return self.track_id_counter


class PathPlanningEngine:
    """Path planning and trajectory generation."""
    
    def __init__(self):
        self.current_path = []
        self.waypoints = []
        
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float], 
                 obstacles: List[Dict], vehicle_state: Dict) -> Dict:
        """Plan optimal path from start to goal avoiding obstacles."""
        
        # Simple A* path planning implementation
        path_points = self.a_star_planning(start, goal, obstacles)
        
        # Generate smooth trajectory
        trajectory = self.generate_smooth_trajectory(path_points, vehicle_state)
        
        # Calculate path metrics
        path_length = self.calculate_path_length(path_points)
        estimated_time = self.estimate_travel_time(path_points, vehicle_state)
        
        return {
            'path_points': path_points,
            'trajectory': trajectory,
            'path_length': round(path_length, 2),
            'estimated_time': round(estimated_time, 2),
            'obstacles_avoided': len(obstacles),
            'path_quality': self.assess_path_quality(path_points, obstacles)
        }
    
    def a_star_planning(self, start: Tuple[float, float], goal: Tuple[float, float], 
                       obstacles: List[Dict]) -> List[Tuple[float, float]]:
        """A* path planning algorithm (simplified)."""
        
        # For demonstration, create a simple path avoiding obstacles
        path = [start]
        
        # Simple obstacle avoidance
        current = start
        step_size = 5.0  # meters
        
        while euclidean(current, goal) > step_size:
            # Calculate direction to goal
            dx = goal[0] - current[0]
            dy = goal[1] - current[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Normalize direction
            dx /= distance
            dy /= distance
            
            # Check for obstacles in path
            next_point = (current[0] + dx * step_size, current[1] + dy * step_size)
            
            # Simple obstacle avoidance
            if self.is_point_in_obstacle(next_point, obstacles):
                # Try to go around obstacle
                perpendicular_options = [
                    (current[0] + dy * step_size, current[1] - dx * step_size),  # Left
                    (current[0] - dy * step_size, current[1] + dx * step_size)   # Right
                ]
                
                for option in perpendicular_options:
                    if not self.is_point_in_obstacle(option, obstacles):
                        next_point = option
                        break
            
            path.append(next_point)
            current = next_point
        
        path.append(goal)
        return path
    
    def is_point_in_obstacle(self, point: Tuple[float, float], obstacles: List[Dict]) -> bool:
        """Check if point is inside any obstacle."""
        for obstacle in obstacles:
            # Assume obstacles are circles for simplicity
            obstacle_center = (obstacle.get('x', 0), obstacle.get('y', 0))
            obstacle_radius = obstacle.get('radius', 5)
            
            if euclidean(point, obstacle_center) < obstacle_radius:
                return True
        
        return False
    
    def generate_smooth_trajectory(self, path_points: List[Tuple[float, float]], 
                                 vehicle_state: Dict) -> List[Dict]:
        """Generate smooth trajectory with speed and steering profiles."""
        trajectory = []
        
        current_speed = vehicle_state.get('speed', 10)  # m/s
        max_speed = 30  # m/s
        max_acceleration = 3  # m/s²
        max_deceleration = -5  # m/s²
        
        for i, point in enumerate(path_points):
            # Calculate curvature at this point
            curvature = self.calculate_curvature(path_points, i)
            
            # Adjust speed based on curvature
            speed_limit_curve = min(max_speed, math.sqrt(abs(1.0 / (curvature + 0.01))))
            target_speed = min(max_speed, speed_limit_curve)
            
            # Calculate steering angle
            if i < len(path_points) - 1:
                next_point = path_points[i + 1]
                heading = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
                steering_angle = math.degrees(heading) % 360
            else:
                steering_angle = 0
            
            trajectory.append({
                'x': point[0],
                'y': point[1],
                'speed': round(target_speed, 2),
                'steering_angle': round(steering_angle, 2),
                'curvature': round(curvature, 4)
            })
        
        return trajectory
    
    def calculate_curvature(self, path_points: List[Tuple[float, float]], index: int) -> float:
        """Calculate curvature at given point."""
        if index == 0 or index == len(path_points) - 1:
            return 0.0
        
        # Use three points to estimate curvature
        p1 = path_points[index - 1]
        p2 = path_points[index]
        p3 = path_points[index + 1]
        
        # Calculate area of triangle
        area = 0.5 * abs((p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])))
        
        # Calculate side lengths
        a = euclidean(p1, p2)
        b = euclidean(p2, p3)
        c = euclidean(p3, p1)
        
        # Curvature formula
        if a * b * c > 0:
            curvature = 4 * area / (a * b * c)
        else:
            curvature = 0.0
        
        return curvature
    
    def calculate_path_length(self, path_points: List[Tuple[float, float]]) -> float:
        """Calculate total path length."""
        total_length = 0.0
        
        for i in range(len(path_points) - 1):
            total_length += euclidean(path_points[i], path_points[i + 1])
        
        return total_length
    
    def estimate_travel_time(self, path_points: List[Tuple[float, float]], 
                           vehicle_state: Dict) -> float:
        """Estimate travel time for the path."""
        avg_speed = vehicle_state.get('speed', 15)  # m/s
        path_length = self.calculate_path_length(path_points)
        
        return path_length / avg_speed if avg_speed > 0 else 0


class BehaviorPlanningEngine:
    """High-level behavior planning and decision making."""
    
    def __init__(self):
        self.current_behavior = 'lane_keeping'
        self.behaviors = [
            'lane_keeping', 'lane_change_left', 'lane_change_right',
            'following', 'overtaking', 'merging', 'stopping', 'parking'
        ]
        
    def plan_behavior(self, fused_data: Dict, vehicle_state: Dict, 
                     traffic_rules: Dict = None) -> Dict:
        """Plan high-level driving behavior."""
        
        # Analyze current situation
        situation = self.analyze_driving_situation(fused_data, vehicle_state)
        
        # Determine appropriate behavior
        planned_behavior = self.select_behavior(situation, traffic_rules)
        
        # Generate behavior-specific commands
        commands = self.generate_behavior_commands(planned_behavior, situation)
        
        return {
            'current_behavior': self.current_behavior,
            'planned_behavior': planned_behavior,
            'situation_assessment': situation,
            'commands': commands,
            'confidence': situation.get('confidence', 0.8),
            'safety_status': self.assess_safety(situation)
        }
    
    def analyze_driving_situation(self, fused_data: Dict, vehicle_state: Dict) -> Dict:
        """Analyze current driving situation."""
        fused_objects = fused_data.get('fused_objects', [])
        environment = fused_data.get('environment', {})
        
        # Classify objects by relevance
        lead_vehicle = None
        following_vehicles = []
        adjacent_vehicles = []
        
        for obj in fused_objects:
            if obj['type'] in ['car', 'truck']:
                distance = obj['distance']
                
                # Simple lane assignment (would use more sophisticated method in practice)
                if distance < 50 and abs(obj.get('lateral_offset', 0)) < 2:  # Same lane
                    if obj.get('velocity', 0) < vehicle_state.get('speed', 0):
                        lead_vehicle = obj
                    else:
                        following_vehicles.append(obj)
                else:
                    adjacent_vehicles.append(obj)
        
        # Traffic density
        traffic_density = len(fused_objects) / 10.0  # Normalize
        
        # Speed analysis
        current_speed = vehicle_state.get('speed', 0)
        speed_limit = 25  # m/s (default)
        speed_ratio = current_speed / speed_limit
        
        return {
            'lead_vehicle': lead_vehicle,
            'following_vehicles': following_vehicles,
            'adjacent_vehicles': adjacent_vehicles,
            'traffic_density': min(1.0, traffic_density),
            'current_speed': current_speed,
            'speed_ratio': speed_ratio,
            'weather_condition': environment.get('weather_condition', 'clear'),
            'driving_conditions': environment.get('driving_conditions', 'good'),
            'confidence': fused_data.get('confidence', 0.8)
        }
    
    def select_behavior(self, situation: Dict, traffic_rules: Dict = None) -> str:
        """Select appropriate driving behavior."""
        lead_vehicle = situation.get('lead_vehicle')
        current_speed = situation.get('current_speed', 0)
        speed_ratio = situation.get('speed_ratio', 0)
        traffic_density = situation.get('traffic_density', 0)
        
        # Decision logic
        if lead_vehicle and lead_vehicle['distance'] < 20:
            if lead_vehicle.get('velocity', 0) < current_speed - 5:
                return 'following'
            elif traffic_density < 0.3:
                return 'lane_change_left'  # Attempt overtaking
            else:
                return 'following'
        
        elif speed_ratio < 0.8 and traffic_density < 0.5:
            return 'lane_keeping'  # Can maintain speed
        
        elif traffic_density > 0.7:
            return 'following'  # Dense traffic
        
        else:
            return 'lane_keeping'  # Default behavior
    
    def generate_behavior_commands(self, behavior: str, situation: Dict) -> Dict:
        """Generate specific commands for the planned behavior."""
        commands = {
            'target_speed': 20,  # m/s
            'target_steering': 0,  # degrees
            'target_acceleration': 0,  # m/s²
            'lane_change_signal': None,
            'emergency_brake': False
        }
        
        current_speed = situation.get('current_speed', 0)
        lead_vehicle = situation.get('lead_vehicle')
        
        if behavior == 'lane_keeping':
            commands['target_speed'] = min(25, current_speed + 2)
            commands['target_acceleration'] = 1.0
            
        elif behavior == 'following':
            if lead_vehicle:
                safe_distance = max(10, current_speed * 2)  # 2-second rule
                if lead_vehicle['distance'] < safe_distance:
                    commands['target_speed'] = max(0, current_speed - 3)
                    commands['target_acceleration'] = -2.0
                else:
                    commands['target_speed'] = lead_vehicle.get('velocity', current_speed)
                    commands['target_acceleration'] = 0.5
        
        elif behavior == 'lane_change_left':
            commands['target_steering'] = -5  # degrees
            commands['lane_change_signal'] = 'left'
            commands['target_speed'] = current_speed + 2
            
        elif behavior == 'lane_change_right':
            commands['target_steering'] = 5  # degrees
            commands['lane_change_signal'] = 'right'
            commands['target_speed'] = current_speed + 1
        
        elif behavior == 'stopping':
            commands['target_speed'] = 0
            commands['target_acceleration'] = -3.0
            commands['emergency_brake'] = True
        
        return commands
    
    def assess_safety(self, situation: Dict) -> str:
        """Assess current safety status."""
        lead_vehicle = situation.get('lead_vehicle')
        current_speed = situation.get('current_speed', 0)
        weather = situation.get('weather_condition', 'clear')
        confidence = situation.get('confidence', 1.0)
        
        safety_score = 1.0
        
        # Distance to lead vehicle
        if lead_vehicle:
            safe_distance = current_speed * 2  # 2-second rule
            if lead_vehicle['distance'] < safe_distance * 0.5:
                safety_score -= 0.4  # Critical
            elif lead_vehicle['distance'] < safe_distance:
                safety_score -= 0.2  # Warning
        
        # Weather conditions
        if weather in ['rain', 'fog']:
            safety_score -= 0.2
        
        # Sensor confidence
        if confidence < 0.7:
            safety_score -= 0.3
        
        if safety_score > 0.8:
            return 'safe'
        elif safety_score > 0.5:
            return 'caution'
        else:
            return 'unsafe'


class AutonomousVehicleSystem:
    """Complete autonomous vehicle AI system."""
    
    def __init__(self):
        self.sensor_generator = SensorDataGenerator()
        self.sensor_fusion = SensorFusionEngine()
        self.path_planner = PathPlanningEngine()
        self.behavior_planner = BehaviorPlanningEngine()
        
        self.current_mission = None
        self.system_state = 'idle'
        
    def process_sensor_frame(self, sensor_data: Dict = None) -> Dict:
        """Process a complete sensor data frame."""
        if sensor_data is None:
            # Generate synthetic sensor data
            sensor_data = self.sensor_generator.generate_complete_sensor_frame()
        
        # Sensor fusion
        fused_data = self.sensor_fusion.fuse_sensor_data(sensor_data)
        
        # Behavior planning
        behavior_plan = self.behavior_planner.plan_behavior(
            fused_data, 
            sensor_data.get('vehicle_state', {})
        )
        
        # Path planning (if needed)
        path_plan = None
        if behavior_plan['planned_behavior'] in ['lane_change_left', 'lane_change_right']:
            # Simple path planning for lane change
            current_pos = (0, 0)  # Would use actual GPS position
            target_pos = (100, 3.5 if behavior_plan['planned_behavior'] == 'lane_change_left' else -3.5)
            
            obstacles = []
            for obj in fused_data.get('fused_objects', []):
                if obj['type'] in ['car', 'truck']:
                    # Convert to local coordinates (simplified)
                    obstacles.append({
                        'x': obj['distance'],
                        'y': 0,  # Would calculate actual lateral position
                        'radius': 2.5
                    })
            
            path_plan = self.path_planner.plan_path(
                current_pos, target_pos, obstacles, 
                sensor_data.get('vehicle_state', {})
            )
        
        return {
            'timestamp': sensor_data.get('timestamp', datetime.now().isoformat()),
            'raw_sensors': sensor_data,
            'fused_perception': fused_data,
            'behavior_plan': behavior_plan,
            'path_plan': path_plan,
            'system_health': self.get_system_health(fused_data),
            'control_commands': behavior_plan.get('commands', {})
        }
    
    def get_system_health(self, fused_data: Dict) -> Dict:
        """Get overall system health status."""
        sensor_health = fused_data.get('sensor_health', {})
        confidence = fused_data.get('confidence', 0)
        
        healthy_sensors = sum(1 for status in sensor_health.values() if status == 'healthy')
        total_sensors = len(sensor_health)
        
        system_confidence = healthy_sensors / total_sensors if total_sensors > 0 else 0
        
        if system_confidence > 0.8 and confidence > 0.8:
            status = 'optimal'
        elif system_confidence > 0.6 and confidence > 0.6:
            status = 'good'
        elif system_confidence > 0.4 and confidence > 0.4:
            status = 'degraded'
        else:
            status = 'critical'
        
        return {
            'overall_status': status,
            'sensor_health': sensor_health,
            'system_confidence': round(system_confidence, 3),
            'perception_confidence': confidence,
            'ready_for_autonomous': status in ['optimal', 'good']
        }


# FastAPI Application
app = FastAPI(
    title="Autonomous Vehicle AI API",
    description="Advanced autonomous vehicle system with sensor fusion, path planning, and decision making",
    version="1.0.0"
)

# Global AV system
av_system = AutonomousVehicleSystem()

# Request/Response models
class SensorDataRequest(BaseModel):
    include_lidar: bool = True
    include_camera: bool = True
    include_radar: bool = True
    include_imu: bool = True
    include_gps: bool = True

class PathPlanningRequest(BaseModel):
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float
    obstacles: List[Dict] = []
    max_speed: float = Field(default=25.0, gt=0, le=50)

class AVSystemResponse(BaseModel):
    timestamp: str
    system_health: Dict
    behavior_plan: Dict
    control_commands: Dict
    perception_confidence: float

@app.post("/sensor_data/generate")
async def generate_sensor_data(request: SensorDataRequest):
    """Generate synthetic sensor data."""
    try:
        sensor_data = av_system.sensor_generator.generate_complete_sensor_frame()
        
        # Filter based on request
        filtered_data = {'timestamp': sensor_data['timestamp']}
        
        if request.include_lidar:
            filtered_data['lidar'] = sensor_data['lidar']
        if request.include_camera:
            filtered_data['camera'] = sensor_data['camera']
        if request.include_radar:
            filtered_data['radar'] = sensor_data['radar']
        if request.include_imu:
            filtered_data['imu'] = sensor_data['imu']
        if request.include_gps:
            filtered_data['gps'] = sensor_data['gps']
        
        filtered_data['vehicle_state'] = sensor_data['vehicle_state']
        
        return filtered_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_frame", response_model=AVSystemResponse)
async def process_sensor_frame():
    """Process a complete sensor frame through the AV pipeline."""
    try:
        result = av_system.process_sensor_frame()
        
        return AVSystemResponse(
            timestamp=result['timestamp'],
            system_health=result['system_health'],
            behavior_plan=result['behavior_plan'],
            control_commands=result['control_commands'],
            perception_confidence=result['fused_perception'].get('confidence', 0.8)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/path_planning")
async def plan_path(request: PathPlanningRequest):
    """Plan path from start to goal."""
    try:
        start = (request.start_x, request.start_y)
        goal = (request.goal_x, request.goal_y)
        
        vehicle_state = {
            'speed': request.max_speed * 0.6,  # Conservative speed
            'max_speed': request.max_speed
        }
        
        path_plan = av_system.path_planner.plan_path(
            start, goal, request.obstacles, vehicle_state
        )
        
        return path_plan
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system_status")
async def get_system_status():
    """Get current system status."""
    try:
        # Generate a sensor frame to get current status
        result = av_system.process_sensor_frame()
        
        return {
            "system_status": result['system_health']['overall_status'],
            "autonomous_ready": result['system_health']['ready_for_autonomous'],
            "current_behavior": result['behavior_plan']['current_behavior'],
            "sensor_health": result['system_health']['sensor_health'],
            "perception_confidence": result['fused_perception']['confidence']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Autonomous Vehicle AI API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    logger.info("Starting Autonomous Vehicle AI System")
    
    # Example usage
    system = AutonomousVehicleSystem()
    
    print("\n=== Autonomous Vehicle AI System ===")
    print("Processing sensor frame...")
    
    # Process a complete sensor frame
    result = system.process_sensor_frame()
    
    print("\nSystem Status:")
    print(f"Overall health: {result['system_health']['overall_status']}")
    print(f"Ready for autonomous: {result['system_health']['ready_for_autonomous']}")
    print(f"Current behavior: {result['behavior_plan']['current_behavior']}")
    print(f"Planned behavior: {result['behavior_plan']['planned_behavior']}")
    
    print("\nPerception Results:")
    print(f"Objects detected: {len(result['fused_perception']['fused_objects'])}")
    print(f"Perception confidence: {result['fused_perception']['confidence']}")
    
    print("\nControl Commands:")
    commands = result['control_commands']
    print(f"Target speed: {commands.get('target_speed', 0)} m/s")
    print(f"Target steering: {commands.get('target_steering', 0)}°")
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()