"""
detect_hallucinations.py - Automated physics-based hallucination detection
Checks model outputs for physical impossibilities and inconsistencies
"""

import re
import json
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

class HallucinationDetector:
    def __init__(self):
        # Physics constraints
        self.MAX_DENSITY = 100  # vehicles/km
        self.MAX_SPEED_CITY = 60  # km/h
        self.MAX_SPEED_HIGHWAY = 130  # km/h
        self.ROAD_CAPACITY = 2000  # vehicles/hour/lane
        
        # Temporal patterns
        self.RUSH_HOURS = [7, 8, 9, 17, 18, 19]
        self.NIGHT_HOURS = [0, 1, 2, 3, 4, 5]
        self.WEEKEND_DAYS = [5, 6]  # Sat, Sun
        
        # Violation counters
        self.violation_counts = defaultdict(int)
        
    def detect_all_violations(self, 
                            explanation: str, 
                            metadata: Dict) -> Dict[str, any]:
        """
        Main detection function - checks all types of hallucinations
        
        Args:
            explanation: Model's explanation text
            metadata: Dict with scenario details (timestamp, location, etc.)
        
        Returns:
            Dict with violation flags and details
        """
        violations = {
            'has_violation': False,
            'violation_types': [],
            'details': {}
        }
        
        # 1. Check density violations
        density_viol = self._check_density_violation(explanation, metadata)
        if density_viol['violation']:
            violations['has_violation'] = True
            violations['violation_types'].append('density')
            violations['details']['density'] = density_viol
            self.violation_counts['density'] += 1
        
        # 2. Check speed violations
        speed_viol = self._check_speed_violation(explanation, metadata)
        if speed_viol['violation']:
            violations['has_violation'] = True
            violations['violation_types'].append('speed')
            violations['details']['speed'] = speed_viol
            self.violation_counts['speed'] += 1
        
        # 3. Check temporal inconsistencies
        temporal_viol = self._check_temporal_consistency(explanation, metadata)
        if temporal_viol['violation']:
            violations['has_violation'] = True
            violations['violation_types'].append('temporal')
            violations['details']['temporal'] = temporal_viol
            self.violation_counts['temporal'] += 1
        
        # 4. Check causality violations
        causality_viol = self._check_causality(explanation, metadata)
        if causality_viol['violation']:
            violations['has_violation'] = True
            violations['violation_types'].append('causality')
            violations['details']['causality'] = causality_viol
            self.violation_counts['causality'] += 1
        
        # 5. Check spatial anomalies
        spatial_viol = self._check_spatial_consistency(explanation, metadata)
        if spatial_viol['violation']:
            violations['has_violation'] = True
            violations['violation_types'].append('spatial')
            violations['details']['spatial'] = spatial_viol
            self.violation_counts['spatial'] += 1
        
        return violations
    
    def _check_density_violation(self, text: str, metadata: Dict) -> Dict:
        """Check if reported density exceeds physical limits"""
        violation = {'violation': False, 'reason': ''}
        
        # Extract density mentions
        density_patterns = [
            r'(\d+)\s*vehicles?\s*per\s*k',
            r'density.*?(\d+)',
            r'(\d+)\s*veh/km'
        ]
        
        densities = []
        for pattern in density_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            densities.extend([int(m) for m in matches])
        
        # Check metadata
        if 'reported_density' in metadata:
            densities.append(metadata['reported_density'])
        
        # Check for violations
        for density in densities:
            if density > self.MAX_DENSITY:
                violation['violation'] = True
                violation['reason'] = f"Density {density} veh/km exceeds max {self.MAX_DENSITY}"
                violation['reported_value'] = density
                violation['threshold'] = self.MAX_DENSITY
                break
        
        return violation
    
    def _check_speed_violation(self, text: str, metadata: Dict) -> Dict:
        """Check if reported speeds are physically possible"""
        violation = {'violation': False, 'reason': ''}
        
        # Extract speed mentions
        speed_patterns = [
            r'(\d+)\s*km/?h',
            r'speed.*?(\d+)',
            r'(\d+)\s*kilometers? per hour'
        ]
        
        speeds = []
        for pattern in speed_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            speeds.extend([int(m) for m in matches])
        
        # Check metadata
        if 'reported_speed' in metadata:
            speeds.append(metadata['reported_speed'])
        
        # Determine road type
        road_type = metadata.get('road_type', 'city')
        max_speed = self.MAX_SPEED_HIGHWAY if road_type == 'highway' else self.MAX_SPEED_CITY
        
        # Check for violations
        for speed in speeds:
            if speed > max_speed:
                violation['violation'] = True
                violation['reason'] = f"Speed {speed} km/h exceeds {road_type} limit {max_speed}"
                violation['reported_value'] = speed
                violation['threshold'] = max_speed
                break
        
        return violation
    
    def _check_temporal_consistency(self, text: str, metadata: Dict) -> Dict:
        """Check if temporal patterns match the time"""
        violation = {'violation': False, 'reason': ''}
        
        # Parse timestamp
        timestamp = metadata.get('timestamp', '')
        if not timestamp:
            return violation
        
        try:
            dt = datetime.fromisoformat(timestamp)
            hour = dt.hour
            weekday = dt.weekday()
        except:
            return violation
        
        text_lower = text.lower()
        
        # Check 1: Rush hour claims
        if 'rush hour' in text_lower or 'peak hour' in text_lower:
            if hour not in self.RUSH_HOURS:
                violation['violation'] = True
                violation['reason'] = f"Claims rush hour at {hour}:00 (not rush hour time)"
                violation['claimed_pattern'] = 'rush_hour'
                violation['actual_hour'] = hour
        
        # Check 2: Night traffic claims
        if hour in self.NIGHT_HOURS:
            if any(term in text_lower for term in ['heavy', 'congested', 'rush']):
                violation['violation'] = True
                violation['reason'] = f"Claims heavy traffic at {hour}:00 (night time)"
                violation['claimed_pattern'] = 'heavy_traffic'
                violation['actual_hour'] = hour
        
        # Check 3: Weekend/weekday confusion
        is_weekend = weekday in self.WEEKEND_DAYS
        weekend_terms = ['weekend', 'saturday', 'sunday', 'leisure']
        weekday_terms = ['weekday', 'commute', 'work day']
        
        has_weekend_terms = any(term in text_lower for term in weekend_terms)
        has_weekday_terms = any(term in text_lower for term in weekday_terms)
        
        if is_weekend and has_weekday_terms:
            violation['violation'] = True
            violation['reason'] = "Claims weekday pattern on weekend"
            violation['claimed_pattern'] = 'weekday'
            violation['actual_day'] = 'weekend'
        elif not is_weekend and has_weekend_terms:
            violation['violation'] = True
            violation['reason'] = "Claims weekend pattern on weekday"
            violation['claimed_pattern'] = 'weekend'
            violation['actual_day'] = 'weekday'
        
        return violation
    
    def _check_causality(self, text: str, metadata: Dict) -> Dict:
        """Check if claimed causes actually exist"""
        violation = {'violation': False, 'reason': ''}
        
        text_lower = text.lower()
        
        # Check for causal claims
        causal_patterns = [
            'caused by',
            'due to',
            'because of',
            'spillover from',
            'upstream',
            'downstream'
        ]
        
        has_causal_claim = any(pattern in text_lower for pattern in causal_patterns)
        
        if has_causal_claim:
            # Check metadata for actual cause
            if 'upstream_status' in metadata:
                if metadata['upstream_status'] == 'free-flowing':
                    violation['violation'] = True
                    violation['reason'] = "Claims upstream cause but upstream is free-flowing"
                    violation['claimed_cause'] = 'upstream_congestion'
                    violation['actual_state'] = 'free-flowing'
            
            # Check for incident claims
            incident_terms = ['accident', 'incident', 'crash', 'breakdown']
            if any(term in text_lower for term in incident_terms):
                if metadata.get('incident_present', False) == False:
                    violation['violation'] = True
                    violation['reason'] = "Claims incident but none present"
                    violation['claimed_cause'] = 'incident'
                    violation['actual_state'] = 'no_incident'
        
        return violation
    
    def _check_spatial_consistency(self, text: str, metadata: Dict) -> Dict:
        """Check for spatial impossibilities"""
        violation = {'violation': False, 'reason': ''}
        
        text_lower = text.lower()
        
        # Check 1: Isolated congestion
        if metadata.get('surrounding_segments') == 'all_free_flowing':
            congestion_terms = ['congestion', 'jam', 'heavy', 'slow']
            if any(term in text_lower for term in congestion_terms):
                violation['violation'] = True
                violation['reason'] = "Reports congestion but all surrounding segments free"
                violation['pattern'] = 'isolated_congestion'
        
        # Check 2: Wrong direction flow
        if 'direction' in metadata:
            direction_terms = {
                'northbound': ['north', 'upward'],
                'southbound': ['south', 'downward'],
                'eastbound': ['east', 'rightward'],
                'westbound': ['west', 'leftward']
            }
            
            expected = metadata.get('road_direction', '')
            for direction, terms in direction_terms.items():
                if any(term in text_lower for term in terms):
                    if direction != expected and expected:
                        violation['violation'] = True
                        violation['reason'] = f"Reports {direction} on {expected} road"
                        violation['claimed_direction'] = direction
                        violation['actual_direction'] = expected
        
        return violation
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute hallucination detection metrics"""
        total = len(results)
        
        # Count violations
        detected_violations = sum(1 for r in results if r['has_violation'])
        hallucination_rate = detected_violations / total if total > 0 else 0
        
        # Breakdown by type
        type_counts = defaultdict(int)
        for r in results:
            for vtype in r['violation_types']:
                type_counts[vtype] += 1
        
        metrics = {
            'total_samples': total,
            'detected_violations': detected_violations,
            'hallucination_rate': hallucination_rate,
            'violation_breakdown': dict(type_counts),
            'violations_per_sample': detected_violations / total if total > 0 else 0
        }
        
        return metrics
    
    def generate_report(self, results: List[Dict], output_path: str):
        """Generate detailed violation report"""
        metrics = self.compute_metrics(results)
        
        report = {
            'summary': metrics,
            'detailed_violations': results,
            'violation_examples': self._extract_examples(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Hallucination Detection Report")
        print(f"{'='*50}")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Detected violations: {metrics['detected_violations']}")
        print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
        print(f"\nBreakdown by type:")
        for vtype, count in metrics['violation_breakdown'].items():
            print(f"  {vtype}: {count} ({count/metrics['total_samples']:.1%})")
        
        return metrics
    
    def _extract_examples(self, results: List[Dict], n=5) -> Dict:
        """Extract example violations for each type"""
        examples = defaultdict(list)
        
        for r in results:
            for vtype in r['violation_types']:
                if len(examples[vtype]) < n:
                    examples[vtype].append({
                        'violation': r['details'][vtype],
                        'sample_id': r.get('sample_id', 'unknown')
                    })
        
        return dict(examples)

def main():
    """Example usage"""
    detector = HallucinationDetector()
    
    # Load test cases
    test_df = pd.read_csv('data/adversarial/test_cases.csv')
    
    # Simulate model predictions (replace with actual model output)
    results = []
    for _, row in test_df.iterrows():
        metadata = row.to_dict()
        explanation = row['explanation']
        
        # Detect violations
        violations = detector.detect_all_violations(explanation, metadata)
        violations['sample_id'] = row['id']
        results.append(violations)
    
    # Generate report
    detector.generate_report(results, 'results/hallucination_report.json')

if __name__ == "__main__":
    main()
