"""
Scene2Sim: Annotation-Native Minimal AV Simulator

A lightweight, annotation-native simulator for autonomous vehicle research
that enables fast, deterministic replay and systematic perturbation of 
real-world scenarios.
"""

__version__ = "0.1.0"
__author__ = "S M Jubaer"

# Core exports
from .core.scene import Scene, Agent, Trajectory, Waypoint, RoadConfiguration, AgentType
from .core.simulator import ADSimulator, SimulationLog
from .core.perturbations import PerturbationEngine

# IO exports  
from .io.json_adapter import load_scenario

# Metrics exports
from .metrics.safety import SafetyMetrics

__all__ = [
    # Core classes
    'Scene', 'Agent', 'Trajectory', 'Waypoint', 'RoadConfiguration', 'AgentType',
    'ADSimulator', 'SimulationLog', 'PerturbationEngine',
    
    # IO functions
    'load_scenario',
    
    # Metrics
    'SafetyMetrics',
]
