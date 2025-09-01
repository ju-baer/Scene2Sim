"""
Systematic counterfactual operators for scenario perturbations.
"""
import numpy as np
from typing import Dict, Optional, List
from dataclasses import replace

from .scene import Scene, Agent, Waypoint, LaneSection

class PerturbationEngine:
    """Applies systematic perturbations to scenarios."""
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
    
    def temporal_shift(self, scene: Scene, agent_id: str, 
                      delay: float) -> Scene:
        """
        Shift agent's trajectory in time.
        
        Args:
            scene: Original scenario
            agent_id: Target agent ID
            delay: Time delay in seconds (positive = later start)
        """
        new_scene = scene.copy()
        agent = new_scene.agents[agent_id]
        
        # Shift all waypoint times
        new_waypoints = []
        for wp in agent.trajectory.waypoints:
            new_wp = replace(wp, t=wp.t + delay)
            new_waypoints.append(new_wp)
        
        agent.trajectory.waypoints = new_waypoints
        return new_scene
    
    def speed_scaling(self, scene: Scene, agent_id: str,
                     scale_factor: float) -> Scene:
        """
        Scale agent's speed profile.
        
        Args:
            scene: Original scenario  
            agent_id: Target agent ID
            scale_factor: Speed multiplier (1.0 = no change)
        """
        new_scene = scene.copy()
        agent = new_scene.agents[agent_id]
        
        # Scale speeds and recompute timing
        new_waypoints = [agent.trajectory.waypoints[0]]  # Keep first waypoint timing
        
        for i in range(1, len(agent.trajectory.waypoints)):
            prev_wp = new_waypoints[-1]
            curr_wp = agent.trajectory.waypoints[i]
            
            # Scale the speed
            new_speed = max(0.1, curr_wp.v * scale_factor)
            
            # Recompute timing based on distance and new speed
            dx = curr_wp.x - prev_wp.x
            dy = curr_wp.y - prev_wp.y
            distance = np.sqrt(dx*dx + dy*dy)
            
            avg_speed = 0.5 * (prev_wp.v * scale_factor + new_speed)
            dt = distance / max(0.1, avg_speed)
            new_time = prev_wp.t + dt
            
            new_wp = replace(curr_wp, t=new_time, v=new_speed)
            new_waypoints.append(new_wp)
        
        agent.trajectory.waypoints = new_waypoints
        return new_scene
    
    def lateral_nudge(self, scene: Scene, agent_id: str,
                     offset: float) -> Scene:
        """
        Apply lateral position offset to agent trajectory.
        
        Args:
            scene: Original scenario
            agent_id: Target agent ID  
            offset: Lateral offset in meters (positive = left)
        """
        new_scene = scene.copy()
        agent = new_scene.agents[agent_id]
        
        new_waypoints = []
        for wp in agent.trajectory.waypoints:
            new_wp = replace(wp, y=wp.y + offset)
            new_waypoints.append(new_wp)
            
        agent.trajectory.waypoints = new_waypoints
        return new_scene
    
    def random_perturbation(self, scene: Scene, agent_id: str,
                           time_std: float = 0.5,
                           speed_std: float = 0.2,
                           lateral_std: float = 0.3) -> Scene:
        """
        Apply random perturbation with specified noise levels.
        """
        # Random time shift
        delay = self.rng.normal(0, time_std)
        scene = self.temporal_shift(scene, agent_id, delay)
        
        # Random speed scaling
        scale = self.rng.lognormal(0, speed_std)
        scene = self.speed_scaling(scene, agent_id, scale)
        
        # Random lateral nudge  
        offset = self.rng.normal(0, lateral_std)
        scene = self.lateral_nudge(scene, agent_id, offset)
        
        return scene
    
    def generate_perturbation_batch(self, scene: Scene, 
                                   agent_id: str,
                                   n_samples: int = 100,
                                   **perturbation_params) -> List[Scene]:
        """Generate batch of perturbed scenarios."""
        scenarios = []
        for i in range(n_samples):
            perturbed = self.random_perturbation(scene, agent_id, **perturbation_params)
            perturbed.id = f"{scene.id}_pert_{i:03d}"
            scenarios.append(perturbed)
        return scenarios
