"""
Core data structures for AD-SimLite's internal representation.
Annotation-native design: minimal transforms from dataset labels.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from enum import Enum

class AgentType(Enum):
    EGO = "ego"
    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"

class LaneSection(Enum):
    LEFT = "LEFT"
    MIDDLE = "MIDDLE"
    RIGHT = "RIGHT"

@dataclass
class Waypoint:
    """Single trajectory point with timestamp."""
    t: float  # time (s)
    x: float  # longitudinal position (m)
    y: float  # lateral position (m) 
    v: float  # desired speed (m/s)
    
@dataclass
class Trajectory:
    """Time-parameterized path for an agent."""
    waypoints: List[Waypoint] = field(default_factory=list)
    
    def sample_at(self, t: float) -> Tuple[float, float, float]:
        """Interpolate (x, y, v) at time t."""
        if not self.waypoints:
            return 0.0, 0.0, 0.0
            
        times = [wp.t for wp in self.waypoints]
        
        if t <= times[0]:
            wp = self.waypoints[0]
            return wp.x, wp.y, wp.v
        if t >= times[-1]:
            wp = self.waypoints[-1] 
            return wp.x, wp.y, wp.v
            
        # Linear interpolation between waypoints
        idx = np.searchsorted(times, t) - 1
        wp1, wp2 = self.waypoints[idx], self.waypoints[idx + 1]
        dt = wp2.t - wp1.t
        if dt <= 1e-9:
            return wp1.x, wp1.y, wp1.v
            
        alpha = (t - wp1.t) / dt
        x = wp1.x + alpha * (wp2.x - wp1.x)
        y = wp1.y + alpha * (wp2.y - wp1.y)  
        v = wp1.v + alpha * (wp2.v - wp1.v)
        return x, y, v

@dataclass  
class Agent:
    """Simulation agent with shape and trajectory."""
    id: str
    agent_type: AgentType
    trajectory: Trajectory
    length: float = 4.5  # vehicle length (m)
    width: float = 2.0   # vehicle width (m)
    
    def get_state_at(self, t: float) -> Dict[str, float]:
        """Get agent state at time t."""
        x, y, v = self.trajectory.sample_at(t)
        return {
            'x': x, 'y': y, 'v': v,
            'length': self.length, 'width': self.width
        }

@dataclass
class RoadConfiguration:
    """Road geometry and lane structure."""
    width: float  # total road width (m)
    n_ego_lanes: int  # lanes in ego direction 
    n_opposite_lanes: int  # lanes in opposite direction
    
    @property
    def lane_width(self) -> float:
        """Width of each lane."""
        return self.width / (self.n_ego_lanes + self.n_opposite_lanes)
    
    def get_lane_center_y(self, lane_id: int) -> float:
        """Get lateral center position of lane.
        
        Lane indexing:
        - Ego lanes: 0, 1, 2, ... (0 is rightmost ego lane)
        - Opposite lanes: -1, -2, -3, ... (from road center)
        """
        lw = self.lane_width
        road_center = 0.0
        
        if lane_id >= 0:
            # Ego direction lanes (positive y = left from ego perspective)
            return road_center + (lane_id + 0.5) * lw
        else:
            # Opposite direction lanes (negative y = right from ego)  
            return road_center - (abs(lane_id) - 0.5) * lw
    
    def get_section_offset(self, section: LaneSection) -> float:
        """Get lateral offset within lane section."""
        half_lane = self.lane_width * 0.5
        offsets = {
            LaneSection.LEFT: 0.25 * half_lane,
            LaneSection.MIDDLE: 0.0,
            LaneSection.RIGHT: -0.25 * half_lane
        }
        return offsets[section]

@dataclass
class Scene:
    """Complete simulation scenario."""
    id: str
    road: RoadConfiguration
    agents: Dict[str, Agent] = field(default_factory=dict)
    duration: float = 20.0  # scenario duration (s)
    
    def add_agent(self, agent: Agent) -> None:
        """Add agent to scene."""
        self.agents[agent.id] = agent
    
    def get_ego_agent(self) -> Optional[Agent]:
        """Get the ego vehicle agent."""
        for agent in self.agents.values():
            if agent.agent_type == AgentType.EGO:
                return agent
        return None
    
    def copy(self) -> 'Scene':
        """Create deep copy for perturbations."""
        import copy
        return copy.deepcopy(self)
