"""
Adapter for loading scenarios from JSON annotations.
"""
import json
from typing import Dict, List
from pathlib import Path
import numpy as np

from ..core.scene import (Scene, Agent, Trajectory, Waypoint, 
                         RoadConfiguration, AgentType, LaneSection)

class JSONAdapter:
    """Load scenarios from JSON dataset format."""
    
    SECTION_MAP = {
        "LEFT": LaneSection.LEFT,
        "MIDDLE": LaneSection.MIDDLE, 
        "RIGHT": LaneSection.RIGHT
    }
    
    @classmethod
    def load_from_file(cls, filepath: str, scenario_id: str) -> Scene:
        """Load specific scenario from JSON file."""
        data = json.loads(Path(filepath).read_text())
        
        # Find target scenario
        scenario_data = None
        if isinstance(data, list):
            scenario_data = next((s for s in data if s["id"] == scenario_id), None)
        elif isinstance(data, dict) and data.get("id") == scenario_id:
            scenario_data = data
        
        if scenario_data is None:
            raise ValueError(f"Scenario '{scenario_id}' not found in {filepath}")
        
        return cls._parse_scenario(scenario_data)
    
    @classmethod  
    def _parse_scenario(cls, data: Dict) -> Scene:
        """Parse single scenario from JSON data."""
        # Parse road configuration
        road_config = data["roadConfiguration"]
        road = RoadConfiguration(
            width=road_config["roadWidth"],
            n_ego_lanes=road_config["nEgoDirectionLanes"], 
            n_opposite_lanes=road_config["nEgoOppositeDirectionLanes"]
        )
        
        # Create scene
        scene = Scene(id=data["id"], road=road)
        
        # Parse ego vehicle
        ego_config = data["egoConfiguration"]
        ego_lane = ego_config["egoLaneWrtCenter"]
        ego_y = road.get_lane_center_y(ego_lane)
        
        # Simple ego trajectory: straight line at constant speed
        ego_speed_start = ego_config["egoSpeedStart"]
        ego_speed_end = ego_config.get("egoSpeedEnd", ego_speed_start)
        
        # Create ego trajectory over scenario duration
        duration = 20.0  # default
        ego_waypoints = [
            Waypoint(t=0.0, x=0.0, y=ego_y, v=ego_speed_start),
            Waypoint(t=duration, x=ego_speed_start * duration, y=ego_y, v=ego_speed_end)
        ]
        
        ego_agent = Agent(
            id="ego",
            agent_type=AgentType.EGO,
            trajectory=Trajectory(waypoints=ego_waypoints),
            length=4.5,
            width=2.0
        )
        scene.add_agent(ego_agent)
        
        # Parse pedestrian path
        if "path" in data and data["path"]:
            ped_trajectory = cls._parse_pedestrian_path(data["path"], road)
            
            ped_agent = Agent(
                id="ped_0",
                agent_type=AgentType.PEDESTRIAN, 
                trajectory=ped_trajectory,
                length=0.6,
                width=0.6
            )
            scene.add_agent(ped_agent)
        
        return scene
    
    @classmethod
    def _parse_pedestrian_path(cls, path_data: List[Dict], 
                              road: RoadConfiguration) -> Trajectory:
        """Parse pedestrian waypoints from path data.""" 
        waypoints = []
        current_time = 0.0
        
        for i, waypoint_data in enumerate(path_data):
            # Parse location
            location = waypoint_data["location"]
            lane_id = location["laneId"]
            section = cls.SECTION_MAP[location["laneSection"]]
            distance_to_initial_ego = location["distanceToInitialEgo"]
            
            # Compute global position
            x = distance_to_initial_ego
            y = road.get_lane_center_y(lane_id) + road.get_section_offset(section)
            
            # Parse behavior
            behavior = waypoint_data["behavior"]
            speed = max(0.1, behavior["speed"])  # Ensure positive speed
            
            # Compute timing (except for first waypoint)
            if i > 0:
                prev_wp = waypoints[-1]
                dx = x - prev_wp.x
                dy = y - prev_wp.y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Use average speed for timing
                avg_speed = 0.5 * (speed + prev_wp.v)
                dt = distance / max(0.1, avg_speed)
                current_time += dt
            
            waypoint = Waypoint(t=current_time, x=x, y=y, v=speed)
            waypoints.append(waypoint)
        
        return Trajectory(waypoints=waypoints)

# Convenience function
def load_scenario(filepath: str, scenario_id: str) -> Scene:
    """Load scenario from JSON file."""
    return JSONAdapter.load_from_file(filepath, scenario_id)
