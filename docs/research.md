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

## Statistical Analysis

### Effect Size Calculation
```python
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (n2 - 1) * np.var(group2)) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Compare baseline vs perturbed scenarios
baseline_ttc = [result['baseline_ttc'] for result in baseline_results]
perturbed_ttc = [result['perturbed_ttc'] for result in perturbed_results]

effect_size = cohens_d(baseline_ttc, perturbed_ttc)
print(f"Effect size (Cohen's d): {effect_size:.3f}")

# Interpretation
if abs(effect_size) < 0.2:
    print("Small effect")
elif abs(effect_size) < 0.5:
    print("Medium effect")
else:
    print("Large effect")
```

### Confidence Intervals
```python
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    """Calculate bootstrap confidence interval."""
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(statistic(sample))
    
    lower = np.percentile(bootstrap_samples, 100 * alpha/2)
    upper = np.percentile(bootstrap_samples, 100 * (1 - alpha/2))
    
    return statistic(data), (lower, upper)

# Example usage
collision_rates = [result['collision_rate'] for result in monte_carlo_results]
mean_rate, ci = bootstrap_ci(collision_rates)
print(f"Collision rate: {mean_rate:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
```

## Publication Workflows

### Data Management
```python
# Create reproducible data package
data_package = {
    'metadata': {
        'title': 'Scene2Sim Safety Evaluation Study',
        'authors': ['Author 1', 'Author 2'],
        'date': '2024-01-15',
        'version': '1.0.0',
        'description': 'Systematic safety evaluation using Scene2Sim'
    },
    'scenarios': scenario_ids,
    'parameters': experiment_parameters,
    'results': results_summary,
    'code_version': Scene2Sim.__version__,
    'random_seeds': random_seeds_used
}

# Save reproducible package
import json
with open('study_data_package.json', 'w') as f:
    json.dump(data_package, f, indent=2)
```

### Figure Generation
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def create_publication_figure(results):
    """Create publication-ready figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: TTC distribution
    ttc_values = [r['min_ttc'] for r in results]
    axes[0,0].hist(ttc_values, bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Time to Collision (s)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('TTC Distribution')
    
    # Plot 2: Safety vs perturbation magnitude
    perturbation_magnitudes = [r['perturbation_magnitude'] for r in results]
    safety_scores = [r['safety_score'] for r in results]
    axes[0,1].scatter(perturbation_magnitudes, safety_scores, alpha=0.6)
    axes[0,1].set_xlabel('Perturbation Magnitude')
    axes[0,1].set_ylabel('Safety Score')
    axes[0,1].set_title('Robustness Analysis')
    
    # Plot 3: Parameter sensitivity heatmap
    sensitivity_matrix = create_sensitivity_matrix(results)
    im = axes[1,0].imshow(sensitivity_matrix, cmap='RdYlBu_r')
    axes[1,0].set_title('Parameter Sensitivity')
    plt.colorbar(im, ax=axes[1,0])
    
    # Plot 4: Comparison with baselines
    methods = ['Baseline', 'Method A', 'Method B', 'AD-SimLite']
    scores = [0.85, 0.78, 0.82, 0.91]
    axes[1,1].bar(methods, scores)
    axes[1,1].set_ylabel('Safety Score')
    axes[1,1].set_title('Method Comparison')
    
    plt.tight_layout()
    plt.savefig('publication_figure.pdf', dpi=300, bbox_inches='tight')
    return fig

# Create figure
fig = create_publication_figure(study_results)
```

### Performance Reporting
```python
def generate_performance_report(results, baseline_method='CARLA'):
    """Generate standardized performance report."""
    
    report = {
        'simulation_performance': {
            'avg_simulation_time': np.mean([r['simulation_time'] for r in results]),
            'speedup_vs_realtime': calculate_speedup(results),
            'memory_usage_mb': measure_memory_usage(),
            'cpu_utilization': measure_cpu_usage()
        },
        'accuracy_metrics': {
            'trajectory_fidelity': calculate_trajectory_correlation(results),
            'safety_metric_correlation': correlate_with_baseline(results, baseline_method),
            'behavioral_realism_score': assess_behavioral_realism(results)
        },
        'scalability': {
            'max_agents_tested': max([r['n_agents'] for r in results]),
            'performance_scaling': fit_scaling_curve(results),
            'memory_scaling': analyze_memory_scaling(results)
        }
    }
    
    return report

# Generate and save report
performance_report = generate_performance_report(benchmark_results)
with open('performance_report.json', 'w') as f:
    json.dump(performance_report, f, indent=2)
```

## Ethical Considerations

### Safety & Responsibility
- Always clearly state limitations of simplified 2D model
- Include disclaimers about real-world applicability
- Validate critical findings with higher-fidelity simulations
- Consider societal implications of AV safety research

### Data Privacy
- Ensure dataset usage complies with original licenses
- Anonymize any personally identifiable information
- Respect data provider guidelines and attribution requirements

### Open Science
- Share code and data whenever possible
- Use version control for reproducibility
- Document all assumptions and limitations
- Provide clear installation and usage instructions

---
