#!/usr/bin/env python3
"""
Basic AD-SimLite demonstration.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adsimlite.io.json_adapter import load_scenario
from adsimlite.core.simulator import ADSimulator
from adsimlite.core.perturbations import PerturbationEngine

def main():
    """Run basic demonstration."""
    
    # Load scenario
    print("Loading scenario...")
    try:
        scene = load_scenario("examples/scenarios.json", "psi-0001")
        print(f"Loaded scenario '{scene.id}' with {len(scene.agents)} agents")
    except Exception as e:
        print(f"Error loading scenario: {e}")
        return
    
    # Run baseline simulation
    print("\nRunning baseline simulation...")
    sim = ADSimulator(scene, dt=0.05)
    log = sim.run(headless=True)
    
    print(f"Simulation completed in {log.duration:.3f}s")
    print("Safety metrics:")
    for key, value in log.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Run with perturbations
    print("\nTesting perturbations...")
    perturb_engine = PerturbationEngine()
    
    # Time delay perturbation
    delayed_scene = perturb_engine.temporal_shift(scene, "ped_0", delay=1.0)
    delayed_log = ADSimulator(delayed_scene).run(headless=True)
    
    print(f"With 1s delay - Collisions: {delayed_log.metrics.get('n_collisions', 0)}")
    
    # Speed scaling perturbation  
    fast_scene = perturb_engine.speed_scaling(scene, "ped_0", scale_factor=1.5)
    fast_log = ADSimulator(fast_scene).run(headless=True)
    
    print(f"With 1.5x speed - Collisions: {fast_log.metrics.get('n_collisions', 0)}")
    
    # Export results
    print("\nExporting results...")
    try:
        df = log.to_dataframe()
        df.to_csv("baseline_simulation.csv", index=False)
        print("Saved baseline_simulation.csv")
    except ImportError:
        print("Install pandas to export CSV results")

if __name__ == "__main__":
    main()
