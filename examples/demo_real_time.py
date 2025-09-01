#!/usr/bin/env python3
"""
Real-time simulation demo with live parameter adjustment.
Demonstrates interactive perturbation testing.
"""
import sys
import time
import threading
from pathlib import Path
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adsimlite import load_scenario, ADSimulator
from adsimlite.core.perturbations import PerturbationEngine

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available - running headless demo")

class RealTimeSimulationDemo:
    """Interactive real-time simulation with live parameter adjustment."""
    
    def __init__(self, scenario_file, scenario_id):
        self.base_scene = load_scenario(scenario_file, scenario_id)
        self.perturb_engine = PerturbationEngine(random_seed=42)
        
        # Simulation parameters
        self.current_delay = 0.0
        self.current_speed_scale = 1.0
        self.current_lateral_offset = 0.0
        
        # Control flags
        self.running = True
        self.paused = False
        self.auto_restart = True
        
        # Results tracking
        self.simulation_count = 0
        self.collision_count = 0
        self.results_history = []
        
    def create_current_scenario(self):
        """Create scenario with current perturbation parameters."""
        scene = self.base_scene.copy()
        
        # Apply perturbations
        if abs(self.current_delay) > 0.01:
            scene = self.perturb_engine.temporal_shift(scene, "ped_0", self.current_delay)
        
        if abs(self.current_speed_scale - 1.0) > 0.01:
            scene = self.perturb_engine.speed_scaling(scene, "ped_0", self.current_speed_scale)
            
        if abs(self.current_lateral_offset) > 0.01:
            scene = self.perturb_engine.lateral_nudge(scene, "ped_0", self.current_lateral_offset)
        
        return scene
    
    def run_single_simulation(self):
        """Run one simulation with current parameters."""
        scene = self.create_current_scenario()
        
        if PYGAME_AVAILABLE and not self.paused:
            # Interactive simulation
            sim = ADSimulator(scene, dt=0.05)
            try:
                sim.run_interactive(fps=20)
            except KeyboardInterrupt:
                self.running = False
        else:
            # Headless simulation
            sim = ADSimulator(scene, dt=0.05)
            log = sim.run(headless=True)
            
            # Track results
            self.simulation_count += 1
            if log.metrics['n_collisions'] > 0:
                self.collision_count += 1
            
            self.results_history.append({
                'sim_id': self.simulation_count,
                'delay': self.current_delay,
                'speed_scale': self.current_speed_scale,
                'lateral_offset': self.current_lateral_offset,
                'is_safe': log.metrics['is_safe'],
                'n_collisions': log.metrics['n_collisions'],
                'min_ttc': log.metrics['min_ttc'],
                'min_distance': log.metrics['min_distance_overall']
            })
            
            self.print_simulation_results(log)
    
    def print_simulation_results(self, log):
        """Print current simulation results."""
        print(f"\n=== Simulation #{self.simulation_count} ===")
        print(f"Parameters: delay={self.current_delay:+.1f}s, "
              f"speed={self.current_speed_scale:.2f}x, "
              f"lateral={self.current_lateral_offset:+.1f}m")
        
        safety_status = "SAFE" if log.metrics['is_safe'] else "UNSAFE"
        print(f"Result: {safety_status}")
        
        if log.metrics['n_collisions'] > 0:
            print(f"Collisions: {log.metrics['n_collisions']}")
        
        if log.metrics['min_ttc'] < float('inf'):
            print(f"Min TTC: {log.metrics['min_ttc']:.2f}s")
        
        print(f"Min Distance: {log.metrics['min_distance_overall']:.2f}m")
        
        # Running statistics
        collision_rate = self.collision_count / self.simulation_count
        print(f"Overall: {self.simulation_count} runs, {collision_rate:.1%} collision rate")
    
    def interactive_parameter_control(self):
        """Interactive parameter adjustment via keyboard."""
        if not PYGAME_AVAILABLE:
            return
        
        print("\n🎮 INTERACTIVE CONTROLS:")
        print("  Q/A: Time delay ±0.1s")
        print("  W/S: Speed scale ±0.05x") 
        print("  E/D: Lateral offset ±0.1m")
        print("  R:   Reset parameters")
        print("  P:   Pause/Resume")
        print("  ESC: Quit")
        print("  SPACE: Run single simulation")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_q:
                        self.current_delay += 0.1
                    elif event.key == pygame.K_a:
                        self.current_delay -= 0.1
                    elif event.key == pygame.K_w:
                        self.current_speed_scale += 0.05
                    elif event.key == pygame.K_s:
                        self.current_speed_scale = max(0.1, self.current_speed_scale - 0.05)
                    elif event.key == pygame.K_e:
                        self.current_lateral_offset += 0.1
                    elif event.key == pygame.K_d:
                        self.current_lateral_offset -= 0.1
                    elif event.key == pygame.K_r:
                        self.current_delay = 0.0
                        self.current_speed_scale = 1.0
                        self.current_lateral_offset = 0.0
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_SPACE:
                        self.run_single_simulation()
                    
                    # Print current parameters
                    print(f"Parameters: delay={self.current_delay:+.1f}s, "
                          f"speed={self.current_speed_scale:.2f}x, "
                          f"lateral={self.current_lateral_offset:+.1f}m")
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    def automatic_parameter_sweep(self):
        """Automatic parameter space exploration."""
        print("\n🤖 AUTOMATIC PARAMETER SWEEP")
        
        # Define sweep ranges
        delay_range = np.linspace(-2.0, 2.0, 21)
        speed_range = np.linspace(0.5, 1.5, 11)
        
        total_combinations = len(delay_range) * len(speed_range)
        print(f"Testing {total_combinations} parameter combinations...")
        
        for delay in delay_range:
            for speed_scale in speed_range:
                if not self.running:
                    break
                    
                self.current_delay = delay
                self.current_speed_scale = speed_scale
                self.current_lateral_offset = 0.0
                
                self.run_single_simulation()
                
                # Small delay for visualization
                time.sleep(0.1)
        
        self.print_sweep_summary()
    
    def print_sweep_summary(self):
        """Print summary of parameter sweep results."""
        if not self.results_history:
            return
        
        import pandas as pd
        df = pd.DataFrame(self.results_history)
        
        print(f"\n PARAMETER SWEEP SUMMARY")
        print(f"Total simulations: {len(df)}")
        print(f"Collision rate: {(df['n_collisions'] > 0).mean():.1%}")
        
        # Find worst cases
        unsafe_scenarios = df[df['n_collisions'] > 0]
        if len(unsafe_scenarios) > 0:
            worst_case = unsafe_scenarios.iloc[0]
            print(f"Worst case: delay={worst_case['delay']:+.1f}s, "
                  f"speed={worst_case['speed_scale']:.2f}x")
            print(f"  Collisions: {worst_case['n_collisions']}")
            print(f"  TTC: {worst_case['min_ttc']:.2f}s")
        
        # Export results
        df.to_csv('realtime_demo_results.csv', index=False)
        print(f"Results exported to realtime_demo_results.csv")

def main():
    """Main demo function."""
    print("AD-SimLite Real-Time Demo")
    print("=" * 40)
    
    try:
        demo = RealTimeSimulationDemo(
            "examples/scenarios/pedestrian_crossing.json",
            "crossing_001"
        )
        
        print(f"Loaded scenario: {demo.base_scene.id}")
        print(f"Agents: {list(demo.base_scene.agents.keys())}")
        
        # Choose demo mode
        print(f"\nDemo Modes:")
        print(f"1. Interactive parameter control (requires pygame)")
        print(f"2. Automatic parameter sweep")
        print(f"3. Single simulation test")
        
        try:
            choice = input("\nSelect mode (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "3"  # Default to single simulation
        
        if choice == "1" and PYGAME_AVAILABLE:
            # Interactive mode
            pygame.init()
            pygame.display.set_mode((100, 100))  # Minimal display for event handling
            
            # Start parameter control in separate thread
            control_thread = threading.Thread(target=demo.interactive_parameter_control)
            control_thread.daemon = True
            control_thread.start()
            
            # Auto-restart simulations
            while demo.running:
                if not demo.paused:
                    demo.run_single_simulation()
                time.sleep(1.0)
        
        elif choice == "2":
            # Automatic sweep
            demo.automatic_parameter_sweep()
        
        else:
            # Single simulation test
            print(f"\nRunning single simulation test...")
            demo.run_single_simulation()
    
    except FileNotFoundError as e:
        print(f" Error: Could not find scenario file")
        print(f"Make sure 'examples/scenarios/pedestrian_crossing.json' exists")
        print(f"You can create it using the scenarios in the main examples directory")
    
    except KeyboardInterrupt:
        print(f"\nDemo interrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
