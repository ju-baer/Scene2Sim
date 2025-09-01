"""
Core simulation engine with deterministic time-stepping.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from .scene import Scene, Agent
from ..metrics.safety import SafetyMetrics
from ..utils.geometry import check_collision

@dataclass
class SimulationState:
    """State of all agents at one timestep."""
    time: float
    agents: Dict[str, Dict[str, float]]  # agent_id -> state dict
    collisions: List[Tuple[str, str]] = None  # collision pairs
    
    def __post_init__(self):
        if self.collisions is None:
            self.collisions = []

@dataclass 
class SimulationLog:
    """Complete simulation history and metrics."""
    scene_id: str
    states: List[SimulationState]
    metrics: Dict[str, float]
    duration: float  # wall-clock time (s)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame export")
            
        rows = []
        for state in self.states:
            for agent_id, agent_state in state.agents.items():
                row = {
                    'time': state.time,
                    'agent_id': agent_id,
                    **agent_state,
                    'has_collision': any(agent_id in pair for pair in state.collisions)
                }
                rows.append(row)
        return pd.DataFrame(rows)

class ADSimulator:
    """
    Deterministic simulator with annotation-native replay.
    
    Key features:
    - Fixed timestep integration
    - Collision detection 
    - Safety metric computation
    - Headless operation support
    """
    
    def __init__(self, scene: Scene, dt: float = 0.05, 
                 enable_collisions: bool = True, 
                 enable_metrics: bool = True,
                 random_seed: int = 42):
        self.scene = scene
        self.dt = dt
        self.enable_collisions = enable_collisions
        self.enable_metrics = enable_metrics
        
        # Set random seed for deterministic perturbations
        np.random.seed(random_seed)
        
        # Initialize metrics computer
        if enable_metrics:
            self.safety_metrics = SafetyMetrics()
        
    def step(self, t: float) -> SimulationState:
        """Execute one simulation timestep."""
        # Get agent states at time t
        agent_states = {}
        for agent_id, agent in self.scene.agents.items():
            agent_states[agent_id] = agent.get_state_at(t)
        
        # Check for collisions
        collisions = []
        if self.enable_collisions:
            agent_list = list(self.scene.agents.items())
            for i in range(len(agent_list)):
                for j in range(i + 1, len(agent_list)):
                    id1, agent1 = agent_list[i]
                    id2, agent2 = agent_list[j]
                    
                    state1 = agent_states[id1]
                    state2 = agent_states[id2]
                    
                    if check_collision(
                        (state1['x'], state1['y'], state1['length'], state1['width']),
                        (state2['x'], state2['y'], state2['length'], state2['width'])
                    ):
                        collisions.append((id1, id2))
        
        return SimulationState(time=t, agents=agent_states, collisions=collisions)
    
    def run(self, headless: bool = True) -> SimulationLog:
        """
        Run complete simulation.
        
        Args:
            headless: If False, enable visualization
            
        Returns:
            Complete simulation log with metrics
        """
        start_time = time.time()
        
        # Time stepping
        t = 0.0
        states = []
        
        while t <= self.scene.duration:
            state = self.step(t)
            states.append(state)
            t += self.dt
        
        # Compute metrics
        metrics = {}
        if self.enable_metrics:
            metrics = self.safety_metrics.compute_scenario_metrics(states)
        
        duration = time.time() - start_time
        
        return SimulationLog(
            scene_id=self.scene.id,
            states=states,
            metrics=metrics, 
            duration=duration
        )
    
    def run_interactive(self, fps: int = 20):
        """Run with real-time visualization."""
        try:
            from ..render.pygame_viz import PygameRenderer
        except ImportError:
            raise ImportError("pygame required for interactive visualization")
            
        renderer = PygameRenderer(self.scene, fps=fps)
        
        t = 0.0
        clock = renderer.clock
        
        while t <= self.scene.duration and renderer.running:
            state = self.step(t)
            renderer.render(state)
            
            t += self.dt
            clock.tick(fps)
        
        renderer.quit()
