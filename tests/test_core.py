"""
Unit tests for core AD-SimLite functionality.
"""
import pytest
import numpy as np
from adsimlite.core.scene import (
    Scene, Agent, Trajectory, Waypoint, RoadConfiguration, 
    AgentType, LaneSection
)
from adsimlite.core.simulator import ADSimulator
from adsimlite.core.perturbations import PerturbationEngine

class TestRoadConfiguration:
    """Test road geometry and lane mapping."""
    
    def test_lane_width_calculation(self):
        """Test lane width computation."""
        road = RoadConfiguration(width=14.0, n_ego_lanes=2, n_opposite_lanes=2)
        assert road.lane_width == 3.5
    
    def test_ego_lane_centers(self):
        """Test ego direction lane center positions."""
        road = RoadConfiguration(width=14.0, n_ego_lanes=2, n_opposite_lanes=2)
        
        # Lane 0 (rightmost ego lane)
        assert road.get_lane_center_y(0) == 1.75
        
        # Lane 1 (leftmost ego lane) 
        assert road.get_lane_center_y(1) == 5.25
    
    def test_opposite_lane_centers(self):
        """Test opposite direction lane center positions."""
        road = RoadConfiguration(width=14.0, n_ego_lanes=2, n_opposite_lanes=2)
        
        # Lane -1 (closest opposite lane)
        assert road.get_lane_center_y(-1) == -1.75
        
        # Lane -2 (farthest opposite lane)
        assert road.get_lane_center_y(-2) == -5.25
    
    def test_section_offsets(self):
        """Test lane section offset calculation."""
        road = RoadConfiguration(width=14.0, n_ego_lanes=2, n_opposite_lanes=2)
        
        half_lane = road.lane_width * 0.5  # 1.75
        quarter_lane = half_lane * 0.25    # 0.4375
        
        assert road.get_section_offset(LaneSection.LEFT) == quarter_lane
        assert road.get_section_offset(LaneSection.MIDDLE) == 0.0
        assert road.get_section_offset(LaneSection.RIGHT) == -quarter_lane

class TestTrajectory:
    """Test trajectory interpolation."""
    
    def test_trajectory_sampling(self):
        """Test trajectory interpolation at different times."""
        waypoints = [
            Waypoint(t=0.0, x=0.0, y=0.0, v=1.0),
            Waypoint(t=1.0, x=1.0, y=0.0, v=1.0),
            Waypoint(t=2.0, x=2.0, y=1.0, v=1.0),
        ]
        traj = Trajectory(waypoints=waypoints)
        
        # Test at waypoint times
        x, y, v = traj.sample_at(0.0)
        assert (x, y, v) == (0.0, 0.0, 1.0)
        
        x, y, v = traj.sample_at(1.0)
        assert (x, y, v) == (1.0, 0.0, 1.0)
        
        # Test interpolation
        x, y, v = traj.sample_at(0.5)
        assert (x, y, v) == (0.5, 0.0, 1.0)
        
        x, y, v = traj.sample_at(1.5)
        assert (x, y, v) == (1.5, 0.5, 1.0)
    
    def test_trajectory_edge_cases(self):
        """Test trajectory sampling edge cases."""
        waypoints = [
            Waypoint(t=1.0, x=10.0, y=5.0, v=2.0),
            Waypoint(t=2.0, x=20.0, y=5.0, v=2.0),
        ]
        traj = Trajectory(waypoints=waypoints)
        
        # Before first waypoint
        x, y, v = traj.sample_at(0.0)
        assert (x, y, v) == (10.0, 5.0, 2.0)
        
        # After last waypoint  
        x, y, v = traj.sample_at(3.0)
        assert (x, y, v) == (20.0, 5.0, 2.0)

class TestSimulation:
    """Test simulation mechanics."""
    
    def create_test_scene(self):
        """Create simple test scenario."""
        road = RoadConfiguration(width=7.0, n_ego_lanes=1, n_opposite_lanes=1)
        scene = Scene(id="test", road=road, duration=2.0)
        
        # Ego agent (straight line)
        ego_traj = Trajectory(waypoints=[
            Waypoint(t=0.0, x=0.0, y=road.get_lane_center_y(0), v=10.0),
            Waypoint(t=2.0, x=20.0, y=road.get_lane_center_y(0), v=10.0),
        ])
        ego = Agent("ego", AgentType.EGO, ego_traj, length=4.0, width=2.0)
        scene.add_agent(ego)
        
        return scene
    
    def test_simulation_basic(self):
        """Test basic simulation functionality."""
        scene = self.create_test_scene()
        sim = ADSimulator(scene, dt=0.1, enable_metrics=False)
        
        log = sim.run()
        
        # Check simulation ran
        assert len(log.states) > 0
        assert log.scene_id == "test"
        assert log.duration > 0
        
        # Check ego motion
        first_state = log.states[0]
        last_state = log.states[-1]
        
        ego_start = first_state.agents["ego"]
        ego_end = last_state.agents["ego"]
        
        # Ego should move forward
        assert ego_end['x'] > ego_start['x']
        assert ego_end['y'] == ego_start['y']  # Straight line
    
    def test_deterministic_simulation(self):
        """Test simulation determinism."""
        scene = self.create_test_scene()
        
        # Run twice with same seed
        sim1 = ADSimulator(scene, dt=0.1, random_seed=42)
        sim2 = ADSimulator(scene, dt=0.1, random_seed=42)
        
        log1 = sim1.run()
        log2 = sim2.run()
        
        # Results should be identical
        assert len(log1.states) == len(log2.states)
        
        for state1, state2 in zip(log1.states, log2.states):
            assert state1.time == state2.time
            for agent_id in state1.agents:
                s1 = state1.agents[agent_id]
                s2 = state2.agents[agent_id]
                assert s1['x'] == s2['x']
                assert s1['y'] == s2['y']

class TestPerturbations:
    """Test perturbation operators."""
    
    def create_test_scene_with_ped(self):
        """Create test scene with pedestrian.""" 
        road = RoadConfiguration(width=7.0, n_ego_lanes=1, n_opposite_lanes=1)
        scene = Scene(id="test", road=road, duration=5.0)
        
        # Ego
        ego_traj = Trajectory(waypoints=[
            Waypoint(t=0.0, x=0.0, y=road.get_lane_center_y(0), v=10.0),
            Waypoint(t=5.0, x=50.0, y=road.get_lane_center_y(0), v=10.0),
        ])
        ego = Agent("ego", AgentType.EGO, ego_traj)
        scene.add_agent(ego)
        
        # Pedestrian crossing
        ped_traj = Trajectory(waypoints=[
            Waypoint(t=1.0, x=15.0, y=road.get_lane_center_y(-1), v=1.5),
            Waypoint(t=3.0, x=15.0, y=road.get_lane_center_y(0), v=1.5),
        ])
        ped = Agent("ped_0", AgentType.PEDESTRIAN, ped_traj, length=0.6, width=0.6)
        scene.add_agent(ped)
        
        return scene
    
    def test_temporal_shift(self):
        """Test time delay perturbation."""
        scene = self.create_test_scene_with_ped()
        perturb = PerturbationEngine()
        
        # Apply 1 second delay
        delayed = perturb.temporal_shift(scene, "ped_0", delay=1.0)
        
        # Check pedestrian timing shifted
        orig_times = [wp.t for wp in scene.agents["ped_0"].trajectory.waypoints]
        new_times = [wp.t for wp in delayed.agents["ped_0"].trajectory.waypoints]
        
        for orig_t, new_t in zip(orig_times, new_times):
            assert new_t == orig_t + 1.0
    
    def test_speed_scaling(self):
        """Test speed scaling perturbation."""
        scene = self.create_test_scene_with_ped()
        perturb = PerturbationEngine()
        
        # Apply 2x speed scaling
        fast = perturb.speed_scaling(scene, "ped_0", scale_factor=2.0)
        
        # Check speeds scaled
        orig_speeds = [wp.v for wp in scene.agents["ped_0"].trajectory.waypoints]
        new_speeds = [wp.v for wp in fast.agents["ped_0"].trajectory.waypoints]
        
        for orig_v, new_v in zip(orig_speeds, new_speeds):
            assert abs(new_v - orig_v * 2.0) < 1e-6
    
    def test_lateral_nudge(self):
        """Test lateral offset perturbation."""
        scene = self.create_test_scene_with_ped()
        perturb = PerturbationEngine()
        
        # Apply 0.5m left nudge
        nudged = perturb.lateral_nudge(scene, "ped_0", offset=0.5)
        
        # Check positions shifted
        orig_positions = [(wp.x, wp.y) for wp in scene.agents["ped_0"].trajectory.waypoints]
        new_positions = [(wp.x, wp.y) for wp in nudged.agents["ped_0"].trajectory.waypoints]
        
        for (orig_x, orig_y), (new_x, new_y) in zip(orig_positions, new_positions):
            assert new_x == orig_x  # x unchanged
            assert abs(new_y - (orig_y + 0.5)) < 1e-6  # y shifted

if __name__ == "__main__":
    pytest.main([__file__])
