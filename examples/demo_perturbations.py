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
              f"min_ttc={log.metrics['min_ttc']:.2f}s, "
              f"min_dist={log.metrics['min_distance_overall']:.2f}m")
    
    # Find critical delay range
    collision_delays = [r['delay'] for r in results if r['collisions'] > 0]
    if collision_delays:
        print(f"\nCollisions occur with delays: {min(collision_delays):.1f}s to {max(collision_delays):.1f}s")
    else:
        print("\nNo collisions found in tested range")
    
    return results

def analyze_speed_scaling():
    """Analyze effect of speed perturbations."""
    print("\n=== Speed Scaling Analysis ===")
    
    scene = load_scenario("examples/scenarios.json", "psi-0001")
    perturb = PerturbationEngine()
    
    scales = np.linspace(0.5, 2.0, 16)  # 0.5x to 2.0x speed
    results = []
    
    for scale in scales:
        perturbed = perturb.speed_scaling(scene, "ped_0", scale)
        log = ADSimulator(perturbed, dt=0.05).run(headless=True)
        
        results.append({
            'speed_scale': scale,
            'collisions': log.metrics['n_collisions'],
            'min_ttc': log.metrics['min_ttc'],
            'min_distance': log.metrics['min_distance_overall']
        })
        
        print(f"Speed {scale:.2f}x: "
              f"collisions={log.metrics['n_collisions']}, "
              f"min_ttc={log.metrics['min_ttc']:.2f}s, "
              f"min_dist={log.metrics['min_distance_overall']:.2f}m")
    
    # Find critical speed range
    collision_speeds = [r['speed_scale'] for r in results if r['collisions'] > 0]
    if collision_speeds:
        print(f"\nCollisions occur with speeds: {min(collision_speeds):.2f}x to {max(collision_speeds):.2f}x")
    else:
        print("\nNo collisions found in tested range")
    
    return results

def monte_carlo_analysis():
    """Monte Carlo safety analysis with random perturbations."""
    print("\n=== Monte Carlo Analysis ===")
    
    scene = load_scenario("examples/scenarios.json", "psi-0001")
    perturb = PerturbationEngine()
    
    n_samples = 200
    results = []
    
    print(f"Running {n_samples} random perturbations...")
    
    for i in range(n_samples):
        # Random perturbation
        perturbed = perturb.random_perturbation(
            scene, "ped_0",
            time_std=0.5,      # ±0.5s timing noise
            speed_std=0.2,     # ±20% speed noise
            lateral_std=0.3    # ±0.3m lateral noise
        )
        
        log = ADSimulator(perturbed, dt=0.05).run(headless=True)
        
        results.append({
            'run_id': i,
            'collisions': log.metrics['n_collisions'],
            'min_ttc': log.metrics['min_ttc'],
            'min_distance': log.metrics['min_distance_overall'],
            'ttc_events': log.metrics['n_ttc_events']
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{n_samples} runs")
    
    # Analysis
    collision_rate = sum(1 for r in results if r['collisions'] > 0) / len(results)
    ttc_events = sum(r['ttc_events'] for r in results)
    min_distances = [r['min_distance'] for r in results if r['min_distance'] < float('inf')]
    
    print(f"\nMonte Carlo Results:")
    print(f"  Collision rate: {collision_rate:.1%}")
    print(f"  Total TTC events: {ttc_events}")
    print(f"  Mean min distance: {np.mean(min_distances):.2f}m")
    print(f"  Std min distance: {np.std(min_distances):.2f}m")
    
    return results

def export_results(time_results, speed_results, mc_results):
    """Export analysis results to CSV."""
    try:
        import pandas as pd
        
        # Time analysis
        pd.DataFrame(time_results).to_csv("time_delay_analysis.csv", index=False)
        
        # Speed analysis
        pd.DataFrame(speed_results).to_csv("speed_scaling_analysis.csv", index=False)
        
        # Monte Carlo analysis
        pd.DataFrame(mc_results).to_csv("monte_carlo_analysis.csv", index=False)
        
        print(f"\nResults exported to:")
        print(f"  - time_delay_analysis.csv")
        print(f"  - speed_scaling_analysis.csv")
        print(f"  - monte_carlo_analysis.csv")
        
    except ImportError:
        print(f"\nInstall pandas to export CSV results:")
        print(f"  pip install pandas")

def main():
    """Run comprehensive perturbation analysis."""
    print("AD-SimLite Perturbation Analysis Demo")
    print("=====================================")
    
    try:
        # Run analyses
        time_results = analyze_time_delays()
        speed_results = analyze_speed_scaling()
        mc_results = monte_carlo_analysis()
        
        # Export results
        export_results(time_results, speed_results, mc_results)
        
        print(f"\nAnalysis complete! This demonstrates:")
        print(f"  ✓ Systematic perturbation testing")
        print(f"  ✓ Safety metric computation")
        print(f"  ✓ Statistical analysis capabilities")
        print(f"  ✓ Reproducible research workflows")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure scenarios.json exists in examples/ directory")

if __name__ == "__main__":
    main()
