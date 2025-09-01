#!/usr/bin/env python3
"""
Interactive visualization demo with real-time perturbations.
"""
import sys
from pathlib import Path
import pygame

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsimlite.io.json_adapter import load_scenario
from adsimlite.core.simulator import ADSimulator
from adsimlite.core.perturbations import PerturbationEngine

def main():
    """Run interactive demo."""
    
    print("Loading scenario for interactive demo...")
    try:
        scene = load_scenario("examples/scenarios.json", "psi-0001")
        print(f"Loaded '{scene.id}' - Press SPACE to start simulation")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure scenarios.json exists in examples/ directory")
        return
    
    # Run interactive simulation
    try:
        sim = ADSimulator(scene, dt=0.05)
        sim.run_interactive(fps=20)
        
    except ImportError:
        print("Interactive mode requires pygame. Install with:")
        print("pip install pygame")
        
        # Fallback to headless with metrics
        print("\nRunning headless simulation instead...")
        log = sim.run(headless=True)
        
        print(f"\nResults for {scene.id}:")
        print(f"Duration: {log.duration:.3f}s")
        print("Metrics:")
        for key, value in log.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
