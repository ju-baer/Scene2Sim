#!/usr/bin/env python3
"""
Systematic perturbation analysis demo.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from adsimlite.io.json_adapter import load_scenario
from adsimlite.core.simulator import ADSimulator
from adsimlite.core.perturbations import PerturbationEngine

def analyze_time_delays():
    """Analyze effect of temporal perturbations."""
    print("=== Time Delay Analysis ===")
    
    scene = load_scenario("examples/scenarios.json", "psi-0001")
    perturb = PerturbationEngine()
    
    delays = np.linspace(-2.0, 2.0, 21)  # -2s to +2s
    results = []
    
    for delay in delays:
        perturbed = perturb.temporal_shift(scene, "ped_0", delay)
        log = ADSimulator(perturbed, dt=0.05).run(headless=True)
        
        results.append({
            'delay': delay,
            'collisions': log.metrics['n_collisions'],
            'min_ttc': log.metrics['min_ttc'],
            'min_distance': log.metrics['min_distance_overall']
        })
        
        print(f"Delay {delay:+.1f}s: "
              f"collisions={log.metrics['n_collisions']}, "
              f"min_ttc={log.metrics['min_ttc']:.# AD-SimLite: Complete MVP Implementation
