"""
Scene2Sim: Advanced Scene Analysis and Simulation System

A comprehensive toolkit for transforming images and videos into interactive
3D simulations through computer vision and physics-based modeling.
"""

__version__ = "0.1.0"
__author__ = "Scene2Sim Team"

# Core imports
from .core.scene import Scene, SceneObject, Camera
from .core.simulator import Simulator, SimulationResult
from .io.loaders import load_image, load_video, load_scene
from .analysis.scene_understanding import SceneAnalyzer

# Convenience functions
def quick_analyze(image_path: str, **kwargs) -> Scene:
    """Quick scene analysis from image path."""
    return load_image(image_path, analyze=True, **kwargs)

def quick_simulate(scene: Scene, duration: float = 10.0, **kwargs) -> SimulationResult:
    """Quick simulation of a scene."""
    simulator = Simulator(scene, **kwargs)
    return simulator.run(duration)

# Main API exports
__all__ = [
    # Core classes
    "Scene",
    "SceneObject", 
    "Camera",
    "Simulator",
    "SimulationResult",
    "SceneAnalyzer",
    
    # Loading functions
    "load_image",
    "load_video", 
    "load_scene",
    
    # Convenience functions
    "quick_analyze",
    "quick_simulate",
]
