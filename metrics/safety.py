"""
Safety and behavioral metrics for scenario evaluation.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..core.simulator import SimulationState

@dataclass
class TTCEvent:
    """Time-to-collision event."""
    time: float
    agent1: str  
    agent2: str
    ttc: float
    distance: float

class SafetyMetrics:
    """Compute safety metrics from simulation logs."""
    
    def __init__(self, ttc_threshold: float = 3.0,
                 collision_threshold: float = 0.5):
        self.ttc_threshold = ttc_threshold
        self.collision_threshold = collision_threshold
    
    def compute_ttc(self, state1: Dict[str, float], 
                   state2: Dict[str, float]) -> float:
        """
        Compute time-to-collision between two agents.
        Simplified: assumes constant velocities.
        """
        # Relative position and velocity
        dx = state2['x'] - state1['x']
        dy = state2['y'] - state1['y'] 
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Simple relative velocity (assumes agents move in x direction)
        dvx = state2.get('vx', 0) - state1.get('vx', 0)
        
        # TTC calculation
        if abs(dvx) < 1e-6:
            return float('inf')
        
        ttc = distance / abs(dvx) if dvx != 0 else float('inf')
        return ttc if ttc > 0 else float('inf')
    
    def compute_minimum_distance(self, states: List[SimulationState],
                               agent1: str, agent2: str) -> float:
        """Compute minimum distance between two agents over time."""
        min_dist = float('inf')
        
        for state in states:
            if agent1 in state.agents and agent2 in state.agents:
                s1 = state.agents[agent1]
                s2 = state.agents[agent2] 
                
                dx = s2['x'] - s1['x']
                dy = s2['y'] - s1['y']
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def detect_ttc_events(self, states: List[SimulationState]) -> List[TTCEvent]:
        """Detect all TTC events below threshold."""
        events = []
        
        for state in states:
            agent_ids = list(state.agents.keys())
            
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    id1, id2 = agent_ids[i], agent_ids[j]
                    s1, s2 = state.agents[id1], state.agents[id2]
                    
                    ttc = self.compute_ttc(s1, s2)
                    if ttc <= self.ttc_threshold:
                        dx = s2['x'] - s1['x'] 
                        dy = s2['y'] - s1['y']
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        events.append(TTCEvent(
                            time=state.time,
                            agent1=id1,
                            agent2=id2, 
                            ttc=ttc,
                            distance=distance
                        ))
        
        return events
    
    def count_collisions(self, states: List[SimulationState]) -> int:
        """Count total collision events."""
        collisions = set()
        for state in states:
            for pair in state.collisions:
                collisions.add(tuple(sorted(pair)))
        return len(collisions)
    
    def compute_scenario_metrics(self, states: List[SimulationState]) -> Dict[str, float]:
        """Compute comprehensive safety metrics for scenario."""
        if not states:
            return {}
        
        # Basic counts
        n_collisions = self.count_collisions(states)
        ttc_events = self.detect_ttc_events(states)
        
        # Minimum distances between all agent pairs
        agent_ids = list(states[0].agents.keys()) if states else []
        min_distances = {}
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id1, id2 = agent_ids[i], agent_ids[j]
                min_dist = self.compute_minimum_distance(states, id1, id2)
                min_distances[f"{id1}_{id2}"] = min_dist
        
        # Aggregate metrics
        metrics = {
            'n_collisions': n_collisions,
            'n_ttc_events': len(ttc_events),
            'min_ttc': min([e.ttc for e in ttc_events], default=float('inf')),
            'mean_ttc': np.mean([e.ttc for e in ttc_events]) if ttc_events else float('inf'),
            'min_distance_overall': min(min_distances.values()) if min_distances else float('inf'),
            'mean_min_distance': np.mean(list(min_distances.values())) if min_distances else float('inf'),
            'scenario_duration': states[-1].time if states else 0.0,
            'is_collision_free': n_collisions == 0,
            'is_safe': n_collisions == 0 and len(ttc_events) == 0
        }
        
        return metrics
