"""
generate_adversarial.py - Create physics-violating traffic scenarios
Generates 150 adversarial test cases for hallucination detection
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random

class AdversarialTrafficGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Road network constraints
        self.max_speed_city = 60  # km/h
        self.max_speed_highway = 130  # km/h
        self.max_density = 100  # vehicles per km
        self.road_capacity = 2000  # vehicles per hour per lane
        
        # Temporal patterns
        self.rush_hours = [7, 8, 9, 17, 18, 19]
        self.weekend_days = [5, 6]  # Saturday, Sunday
        
    def generate_physics_violations(self, n_samples=50) -> List[Dict]:
        """Generate scenarios that violate physics constraints"""
        violations = []
        
        for i in range(n_samples):
            violation_type = np.random.choice([
                'density_violation',
                'speed_violation', 
                'capacity_violation',
                'causality_violation'
            ])
            
            if violation_type == 'density_violation':
                # Traffic density exceeds physical road capacity
                scenario = {
                    'id': f'phys_viol_{i}',
                    'type': 'physics_violation',
                    'subtype': 'density',
                    'road_segment': f'A{random.randint(100, 999)}',
                    'timestamp': self._random_timestamp(),
                    'reported_density': random.randint(150, 300),  # Impossible!
                    'road_capacity': self.max_density,
                    'explanation': f"Heavy congestion with {random.randint(150, 300)} vehicles per km",
                    'ground_truth': 'IMPOSSIBLE - Exceeds physical road capacity',
                    'hallucination': True
                }
                
            elif violation_type == 'speed_violation':
                # Vehicles moving at impossible speeds
                scenario = {
                    'id': f'phys_viol_{i}',
                    'type': 'physics_violation',
                    'subtype': 'speed',
                    'road_segment': f'B{random.randint(100, 999)}',
                    'timestamp': self._random_timestamp(),
                    'reported_speed': random.randint(150, 250),  # km/h in city!
                    'speed_limit': self.max_speed_city,
                    'explanation': f"Traffic flowing at {random.randint(150, 250)} km/h",
                    'ground_truth': 'IMPOSSIBLE - Exceeds city speed limits',
                    'hallucination': True
                }
                
            elif violation_type == 'capacity_violation':
                # Flow rate exceeds road capacity
                flow_rate = random.randint(3000, 5000)
                scenario = {
                    'id': f'phys_viol_{i}',
                    'type': 'physics_violation',
                    'subtype': 'capacity',
                    'road_segment': f'C{random.randint(100, 999)}',
                    'timestamp': self._random_timestamp(),
                    'reported_flow': flow_rate,
                    'road_capacity': self.road_capacity,
                    'explanation': f"{flow_rate} vehicles per hour on single lane",
                    'ground_truth': 'IMPOSSIBLE - Exceeds lane capacity',
                    'hallucination': True
                }
                
            else:  # causality_violation
                # Effect without cause
                scenario = {
                    'id': f'phys_viol_{i}',
                    'type': 'physics_violation',
                    'subtype': 'causality',
                    'road_segment': f'D{random.randint(100, 999)}',
                    'timestamp': self._random_timestamp(),
                    'congestion_reason': 'upstream spillover',
                    'upstream_status': 'free-flowing',
                    'explanation': "Congestion caused by upstream bottleneck",
                    'ground_truth': 'IMPOSSIBLE - No upstream congestion exists',
                    'hallucination': True
                }
            
            violations.append(scenario)
            
        return violations
    
    def generate_temporal_inconsistencies(self, n_samples=50) -> List[Dict]:
        """Generate scenarios with impossible temporal patterns"""
        inconsistencies = []
        
        for i in range(n_samples):
            inconsistency_type = np.random.choice([
                'rush_hour_mismatch',
                'weekend_weekday_confusion',
                'holiday_pattern_error'
            ])
            
            if inconsistency_type == 'rush_hour_mismatch':
                # Rush hour traffic at 3 AM
                hour = random.choice([2, 3, 4, 5])
                scenario = {
                    'id': f'temp_incons_{i}',
                    'type': 'temporal_inconsistency',
                    'subtype': 'rush_hour_mismatch',
                    'timestamp': self._timestamp_with_hour(hour),
                    'reported_pattern': 'rush_hour',
                    'actual_time': f'{hour}:00 AM',
                    'explanation': f"Heavy rush hour congestion at {hour}:00 AM",
                    'ground_truth': 'IMPOSSIBLE - No rush hour at night',
                    'hallucination': True
                }
                
            elif inconsistency_type == 'weekend_weekday_confusion':
                # Weekend traffic on Tuesday
                weekday = random.choice([0, 1, 2, 3])  # Mon-Thu (0-indexed)
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
                scenario = {
                    'id': f'temp_incons_{i}',
                    'type': 'temporal_inconsistency',
                    'subtype': 'weekend_pattern',
                    'timestamp': self._timestamp_with_weekday(weekday),
                    'reported_pattern': 'weekend_leisure',
                    'actual_day': weekday_names[weekday],
                    'explanation': "Light weekend leisure traffic pattern",
                    'ground_truth': 'INCONSISTENT - Weekday should have different pattern',
                    'hallucination': True
                }
                
            else:  # holiday_pattern_error
                # Holiday traffic on regular Tuesday
                scenario = {
                    'id': f'temp_incons_{i}',
                    'type': 'temporal_inconsistency',
                    'subtype': 'holiday_pattern',
                    'timestamp': self._regular_weekday_timestamp(),
                    'reported_pattern': 'holiday_reduced',
                    'actual_status': 'regular_workday',
                    'explanation': "Reduced traffic due to public holiday",
                    'ground_truth': 'INCONSISTENT - Not a holiday',
                    'hallucination': True
                }
            
            inconsistencies.append(scenario)
            
        return inconsistencies
    
    def generate_spatial_anomalies(self, n_samples=50) -> List[Dict]:
        """Generate scenarios with impossible spatial patterns"""
        anomalies = []
        
        for i in range(n_samples):
            anomaly_type = np.random.choice([
                'isolated_congestion',
                'wrong_direction_flow',
                'spontaneous_traffic'
            ])
            
            if anomaly_type == 'isolated_congestion':
                # Congestion with no trigger
                scenario = {
                    'id': f'spat_anom_{i}',
                    'type': 'spatial_anomaly',
                    'subtype': 'isolated_congestion',
                    'location': f'Segment_{random.randint(100, 999)}',
                    'timestamp': self._random_timestamp(),
                    'congestion_level': random.randint(80, 100),
                    'surrounding_segments': 'all_free_flowing',
                    'explanation': "Severe congestion on this segment",
                    'ground_truth': 'ANOMALOUS - No visible cause for congestion',
                    'hallucination': True
                }
                
            elif anomaly_type == 'wrong_direction_flow':
                # Traffic flowing wrong way
                scenario = {
                    'id': f'spat_anom_{i}',
                    'type': 'spatial_anomaly',
                    'subtype': 'wrong_direction',
                    'location': f'Highway_{random.randint(1, 50)}',
                    'timestamp': self._random_timestamp(),
                    'reported_direction': 'westbound',
                    'road_direction': 'eastbound_only',
                    'explanation': "Traffic moving westbound",
                    'ground_truth': 'IMPOSSIBLE - One-way road violation',
                    'hallucination': True
                }
                
            else:  # spontaneous_traffic
                # Traffic appears from nowhere
                scenario = {
                    'id': f'spat_anom_{i}',
                    'type': 'spatial_anomaly',
                    'subtype': 'spontaneous_appearance',
                    'location': f'Junction_{random.randint(1, 100)}',
                    'timestamp': self._random_timestamp(),
                    'sudden_density_increase': random.randint(0, 80),
                    'inflow_from_adjacent': 0,
                    'explanation': "Sudden traffic density increase",
                    'ground_truth': 'ANOMALOUS - No inflow to explain increase',
                    'hallucination': True
                }
            
            anomalies.append(scenario)
            
        return anomalies
    
    def _random_timestamp(self) -> str:
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        delta = end - start
        random_days = random.randint(0, delta.days)
        random_hours = random.randint(0, 23)
        return (start + timedelta(days=random_days, hours=random_hours)).isoformat()
    
    def _timestamp_with_hour(self, hour: int) -> str:
        start = datetime(2024, 6, 15, hour, 0)
        return start.isoformat()
    
    def _timestamp_with_weekday(self, weekday: int) -> str:
        # Create timestamp for specific weekday
        start = datetime(2024, 6, 3)  # Monday
        target = start + timedelta(days=weekday)
        return target.isoformat()
    
    def _regular_weekday_timestamp(self) -> str:
        # Tuesday, not a holiday
        return datetime(2024, 6, 4, 14, 0).isoformat()
    
    def generate_all(self) -> pd.DataFrame:
        """Generate all adversarial test cases"""
        print("🔧 Generating adversarial test cases...")
        
        physics = self.generate_physics_violations(50)
        temporal = self.generate_temporal_inconsistencies(50)
        spatial = self.generate_spatial_anomalies(50)
        
        all_cases = physics + temporal + spatial
        df = pd.DataFrame(all_cases)
        
        print(f"✅ Generated {len(df)} adversarial test cases")
        print(f"   - Physics violations: {len(physics)}")
        print(f"   - Temporal inconsistencies: {len(temporal)}")
        print(f"   - Spatial anomalies: {len(spatial)}")
        
        return df

def main():
    generator = AdversarialTrafficGenerator()
    
    # Generate all test cases
    adversarial_df = generator.generate_all()
    
    # Save to disk
    adversarial_df.to_csv('data/adversarial/test_cases.csv', index=False)
    adversarial_df.to_json('data/adversarial/test_cases.json', orient='records', indent=2)
    
    # Print sample
    print("\n📊 Sample adversarial case:")
    print(json.dumps(adversarial_df.iloc[0].to_dict(), indent=2))
    
    # Statistics
    print("\n📈 Dataset statistics:")
    print(adversarial_df.groupby(['type', 'subtype']).size())

if __name__ == "__main__":
    main()