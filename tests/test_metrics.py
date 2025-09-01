"""
Unit tests for safety metrics.
"""
import pytest
from adsimlite.metrics.safety import SafetyMetrics, TTCEvent
from adsimlite.core.simulator import SimulationState

class TestSafetyMetrics:
    """Test safety metric calculations."""
    
    def test_ttc_computation(self):
        """Test time-to-collision calculation.""" 
        metrics = SafetyMetrics()
        
        # Head-on collision scenario
        state1 = {'x': 0, 'y': 0, 'vx': 10}  # Moving right at 10 m/s
        state2 = {'x': 50, 'y': 0, 'vx': -10}  # Moving left at 10 m/s
        
        ttc = metrics.compute_ttc(state1, state2)
        expected_ttc = 50 / 20  # distance / relative_speed = 2.5s
        assert abs(ttc - expected_ttc) < 1e-6
    
    def test_minimum_distance_tracking(self):
        """Test minimum distance computation."""
        metrics = SafetyMetrics()
        
        # Create states where agents approach then separate
        states = [
            SimulationState(0.0, {'ego': {'x': 0, 'y': 0}, 'ped': {'x': 10, 'y': 0}}),
            SimulationState(1.0, {'ego': {'x': 5, 'y': 0}, 'ped': {'x': 8, 'y': 0}}),
            SimulationState(2.0, {'ego': {'x': 10, 'y': 0}, 'ped': {'x': 6, 'y': 0}}),  # Closest
            SimulationState(3.0, {'ego': {'x': 15, 'y': 0}, 'ped': {'x': 4, 'y': 0}}),
        ]
        
        min_dist = metrics.compute_minimum_distance(states, 'ego', 'ped')
        assert abs(min_dist - 4.0) < 1e-6  # Minimum distance is 4m
    
    def test_collision_counting(self):
        """Test collision detection and counting."""
        metrics = SafetyMetrics()
        
        states = [
            SimulationState(0.0, {}, collisions=[]),
            SimulationState(1.0, {}, collisions=[('ego', 'ped')]),
            SimulationState(2.0, {}, collisions=[('ego', 'ped')]),  # Same collision continues
            SimulationState(3.0, {}, collisions=[('ego', 'veh'), ('ped', 'veh')]),  # New collisions
        ]
        
        n_collisions = metrics.count_collisions(states)
        assert n_collisions == 3  # ego-ped, ego-veh, ped-veh
    
    def test_scenario_metrics_computation(self):
        """Test comprehensive scenario metrics."""
        metrics = SafetyMetrics()
        
        # Mock states for comprehensive test
        states = [
            SimulationState(0.0, {
                'ego': {'x': 0, 'y': 0, 'vx': 10},
                'ped': {'x': 20, 'y': 0, 'vx': -5}
            }, collisions=[]),
            SimulationState(1.0, {
                'ego': {'x': 10, 'y': 0, 'vx': 10}, 
                'ped': {'x': 15, 'y': 0, 'vx': -5}
            }, collisions=[]),
            SimulationState(2.0, {
                'ego': {'x': 20, 'y': 0, 'vx': 10},
                'ped': {'x': 10, 'y': 0, 'vx': -5}
            }, collisions=[('ego', 'ped')]),
        ]
        
        scenario_metrics = metrics.compute_scenario_metrics(states)
        
        # Check computed metrics
        assert scenario_metrics['n_collisions'] == 1
        assert scenario_metrics['scenario_duration'] == 2.0
        assert scenario_metrics['is_collision_free'] == False
        assert scenario_metrics['is_safe'] == False
        assert scenario_metrics['min_distance_overall'] < float('inf')

if __name__ == "__main__":
    pytest.main([__file__])
