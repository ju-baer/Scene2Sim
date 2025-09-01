# Quick Start Guide

Get AD-SimLite running in 5 minutes with this step-by-step guide.

## Installation

### Option 1: Basic Installation
```bash
pip install Scene2Sim
```

### Option 2: Development Installation
```bash
git clone https://github.com/ju-baer/Scene2Sim.git
cd Scene2Sim
pip install -e ".[full]"
```

### Option 3: Docker
```bash
docker pull Scene2Sim/Scene2Sim:latest
docker run -it Scene2Sim/Scene2Sim:latest
```

## Your First Simulation

### Step 1: Load a Scenario
```python
from Scene2Sim import load_scenario, ADSimulator

# Load from JSON annotations
scene = load_scenario("examples/scenarios/urban_intersection.json", "scenario_001")
print(f"Loaded '{scene.id}' with {len(scene.agents)} agents")
```

### Step 2: Run Simulation
```python
# Create simulator
sim = ADSimulator(scene, dt=0.05)  # 20 FPS

# Run headless simulation
log = sim.run(headless=True)
print(f"Simulation completed in {log.duration:.3f}s")
```

### Step 3: Analyze Results
```python
# Check safety metrics
metrics = log.metrics
print(f"Collisions: {metrics['n_collisions']}")
print(f"Min TTC: {metrics['min_ttc']:.2f}s")
print(f"Safety: {'SAFE' if metrics['is_safe'] else 'UNSAFE'}")

# Export results
df = log.to_dataframe()
df.to_csv("my_first_simulation.csv")
```

## Interactive Visualization

```python
# Real-time visualization
sim.run_interactive(fps=20)

# Controls:
# - Arrow keys: Pan camera
# - Space: Pause/resume
# - ESC: Exit
```

## Your First Perturbation

```python
from Scene2Sim.core.perturbations import PerturbationEngine

# Create perturbation engine
perturb = PerturbationEngine(random_seed=42)

# Apply time delay
delayed_scene = perturb.temporal_shift(scene, "ped_0", delay=1.0)

# Compare results
baseline_log = ADSimulator(scene).run(headless=True)
perturbed_log = ADSimulator(delayed_scene).run(headless=True)

print(f"Baseline collisions: {baseline_log.metrics['n_collisions']}")
print(f"Perturbed collisions: {perturbed_log.metrics['n_collisions']}")
```

## What's Next?

-  [**Basic Simulation Tutorial**](../examples/01_basic_simulation.ipynb)
-  [**Perturbation Analysis**](../examples/02_perturbation_analysis.ipynb)
-  [**Safety Metrics Guide**](../examples/03_safety_metrics.ipynb)
-  [**Research Workflows**](research.md)

---

