# Research Workflows

Best practices for using Scene2Sim in academic research.

## Research Design Principles

### 1. Reproducibility
```python
# Always set random seeds
from Scene2Sim.core.perturbations import PerturbationEngine

perturb = PerturbationEngine(random_seed=42)
sim = ADSimulator(scene, dt=0.05, random_seed=42)

# Version control your scenarios
scenario_config = {
    'dataset_version': 'v1.2.3',
    'scenario_ids': ['urban_001', 'urban_002', 'highway_001'],
    'simulation_params': {'dt': 0.05, 'horizon': 20.0}
}
```

### 2. Statistical Rigor
```python
import numpy as np
from scipy import stats

# Power analysis for sample size
def calculate_sample_size(effect_size, alpha=0.05, power=0.8):
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    return int(2 * ((z_alpha + z_beta) / effect_size)**2)

n_samples = calculate_sample_size(effect_size=0.5)
print(f"Required sample size: {n_samples}")
```

### 3. Systematic Evaluation
```python
# Define parameter space systematically
parameter_space = {
    'time_delays': np.linspace(-3.0, 3.0, 31),
    'speed_scales': np.logspace(-0.3, 0.3, 21),  # 0.5x to 2x
    'lateral_offsets': np.linspace(-1.0, 1.0, 21)
}

# Full factorial design
results = []
for delay in parameter_space['time_delays']:
    for scale in parameter_space['speed_scales']:
        for offset in parameter_space['lateral_offsets']:
            # Run experiment
            result = run_experiment(delay, scale, offset)
            results.append(result)
```

## Experimental Protocols

### Safety Evaluation Protocol
```python
class SafetyEvaluationProtocol:
    """Standardized safety evaluation protocol."""
    
    def __init__(self, scenarios, n_monte_carlo=1000, alpha=0.05):
        self.scenarios = scenarios
        self.n_monte_carlo = n_monte_carlo
        self.alpha = alpha
        self.perturb_engine = PerturbationEngine(random_seed=42)
    
    def evaluate_scenario_safety(self, scenario):
        """Comprehensive safety evaluation."""
        results = {
            'scenario_id': scenario.id,
            'baseline_safety': self._baseline_safety(scenario),
            'perturbation_sensitivity': self._perturbation_sensitivity(scenario),
            'monte_carlo_risk': self._monte_carlo_risk(scenario),
            'critical_parameters': self._find_critical_parameters(scenario)
        }
        return results
    
    def _baseline_safety(self, scenario):
        """Baseline safety metrics."""
        log = ADSimulator(scenario).run(headless=True)
        return {
            'is_safe': log.metrics['is_safe'],
            'collision_free': log.metrics['is_collision_free'],
            'min_ttc': log.metrics['min_ttc'],
            'min_distance': log.metrics['min_distance_overall']
        }
    
    def _perturbation_sensitivity(self, scenario):
        """Test sensitivity to systematic perturbations."""
        sensitivity = {}
        
        # Time sensitivity
        time_results = []
        for delay in np.linspace(-2.0, 2.0, 21):
            perturbed = self.perturb_engine.temporal_shift(scenario, "ped_0", delay)
            log = ADSimulator(perturbed).run(headless=True)
            time_results.append({
                'delay': delay,
                'safe': log.metrics['is_safe'],
                'ttc': log.metrics['min_ttc']
            })
        
        sensitivity['temporal'] = {
            'critical_range': self._find_critical_range(time_results, 'delay'),
            'failure_rate': self._calculate_failure_rate(time_results)
        }
        
        return sensitivity
    
    def _monte_carlo_risk(self, scenario):
        """Monte Carlo risk assessment."""
        failures = 0
        ttc_values = []
        
        for i in range(self.n_monte_carlo):
            perturbed = self.perturb_engine.random_perturbation(
                scenario, "ped_0",
                time_std=0.5, speed_std=0.2, lateral_std=0.3
            )
            log = ADSimulator(perturbed).run(headless=True)
            
            if not log.metrics['is_safe']:
                failures += 1
            ttc_values.append(log.metrics['min_ttc'])
        
        return {
            'failure_rate': failures / self.n_monte_carlo,
            'confidence_interval': self._wilson_score_interval(
                failures, self.n_monte_carlo, self.alpha
            ),
            'ttc_distribution': {
                'mean': np.mean(ttc_values),
                'std': np.std(ttc_values),
                'percentiles': np.percentile(ttc_values, [5, 25, 50, 75, 95])
            }
        }
```

### Planner Benchmarking Protocol
```python
class PlannerBenchmark:
    """Standardized planner evaluation protocol."""
    
    def __init__(self, planner_factory, test_scenarios):
        self.planner_factory = planner_factory
        self.test_scenarios = test_scenarios
    
    def run_benchmark(self, n_runs=100):
        """Run comprehensive planner benchmark."""
        results = []
        
        for scenario in self.test_scenarios:
            scenario_results = self._evaluate_planner_on_scenario(
                scenario, n_runs
            )
            results.append(scenario_results)
        
        return self._aggregate_results(results)
    
    def _evaluate_planner_on_scenario(self, scenario, n_runs):
        """Evaluate planner on single scenario with perturbations."""
        planner = self.planner_factory()
        perturb = PerturbationEngine()
        
        results = []
        for run_id in range(n_runs):
            # Create perturbed scenario
            if run_id == 0:
                test_scenario = scenario  # Baseline
            else:
                test_scenario = perturb.random_perturbation(
                    scenario, "ped_0",
                    time_std=0.3, speed_std=0.1, lateral_std=0.2
                )
            
            # Run planner
            planner_result = self._run_planner(planner, test_scenario)
            results.append({
                'run_id': run_id,
                'is_baseline': run_id == 0,
                **planner_result
            })
        
        return {
            'scenario_id': scenario.id,
            'n_runs': n_runs,
            'results': results,
            'summary': self._summarize_scenario_results(results)
        }
```

---
