# Advanced Usage Guide

Advanced features and customization options for power users and researchers.

## Custom Perturbation Operators

### Creating Custom Perturbations

```python
from adsimlite.core.perturbations import PerturbationEngine
from adsimlite.core.scene import Waypoint
import numpy as np

class CustomPerturbationEngine(PerturbationEngine):
    """Extended perturbation engine with custom operators."""
    
    def sinusoidal_path(self, scene, agent_id, amplitude=0.5, frequency=0.1):
        """Apply sinusoidal trajectory perturbation."""
        new_scene = scene.copy()
        agent = new_scene.agents[agent_id]
        
        new_waypoints = []
        for wp in agent.trajectory.waypoints:
            # Apply sinusoidal lateral offset
            phase = frequency * wp.x
            lateral_offset = amplitude * np.sin(phase)
            
            new_wp = Waypoint(
                t=wp.t,
                x=wp.x,
                y=wp.y + lateral_offset,
                v=wp.v
            )
            new_waypoints.append(new_wp)
        
        agent.trajectory.waypoints = new_waypoints
        return new_scene
    
    def acceleration_burst(self, scene, agent_id, start_time=2.0, 
                          duration=1.0, acceleration=2.0):
        """Apply sudden acceleration burst."""
        new_scene = scene.copy()
        agent = new_scene.agents[agent_id]
        
        new_waypoints = []
        for wp in agent.trajectory.waypoints:
            # Apply acceleration during burst period
            if start_time <= wp.t <= start_time + duration:
                dt = wp.t - start_time
                new_speed = wp.v + acceleration * dt
                new_wp = Waypoint(t=wp.t, x=wp.x, y=wp.y, v=new_speed)
            else:
                new_wp = wp
            
            new_waypoints.append(new_wp)
        
        agent.trajectory.waypoints = new_waypoints
        return new_scene
    
    def visibility_occlusion(self, scene, occluder_position, occluder_size):
        """Add visibility occlusion (conceptual - for planning integration)."""
        # This would integrate with planner perception models
        # For now, we modify scenario metadata
        new_scene = scene.copy()
        
        if not hasattr(new_scene, 'occlusions'):
            new_scene.occlusions = []
        
        new_scene.occlusions.append({
            'position': occluder_position,
            'size': occluder_size,
            'type': 'visibility_block'
        })
        
        return new_scene

# Usage
custom_perturb = CustomPerturbationEngine()
scene_with_sine = custom_perturb.sinusoidal_path(scene, "ped_0", amplitude=1.0)
scene_with_accel = custom_perturb.acceleration_burst(scene, "ped_0", acceleration=3.0)
```
### Probabilistic Perturbations

```python
def probabilistic_perturbation_suite(scene, agent_id, n_samples=100):
    """Generate ensemble of probabilistically perturbed scenarios."""
    
    perturb = PerturbationEngine()
    scenarios = []
    
    for i in range(n_samples):
        # Sample perturbation parameters from distributions
        time_delay = np.random.normal(0, 0.5)  # ±0.5s std
        speed_scale = np.random.lognormal(0, 0.2)  # 20% speed variation
        lateral_offset = np.random.normal(0, 0.3)  # ±30cm std
        
        # Apply combined perturbation
        perturbed = perturb.temporal_shift(scene, agent_id, time_delay)
        perturbed = perturb.speed_scaling(perturbed, agent_id, speed_scale)
        perturbed = perturb.lateral_nudge(perturbed, agent_id, lateral_offset)
        
        perturbed.id = f"{scene.id}_prob_{i:03d}"
        scenarios.append(perturbed)
    
    return scenarios

# Monte Carlo analysis with probabilistic perturbations
prob_scenarios = probabilistic_perturbation_suite(base_scene, "ped_0", 1000)
results = []

for scenario in prob_scenarios:
    log = ADSimulator(scenario).run(headless=True)
    results.append({
        'scenario_id': scenario.id,
        'is_safe': log.metrics['is_safe'],
        'collision_rate': 1 if log.metrics['n_collisions'] > 0 else 0
    })

# Statistical analysis
import pandas as pd
df = pd.DataFrame(results)
print(f"Monte Carlo collision rate: {df['collision_rate'].mean():.1%}")
print(f"95% confidence interval: {df['collision_rate'].std() * 1.96:.1%}")
```

---

## Advanced Metrics
### Custom Safety Metrics
```python
from adsimlite.metrics.safety import SafetyMetrics
import numpy as np

class AdvancedSafetyMetrics(SafetyMetrics):
    """Extended safety metrics with domain-specific measures."""
    
    def compute_drac(self, ego_states, other_states):
        """Deceleration Rate to Avoid Crash (DRAC)."""
        dracs = []
        
        for i in range(len(ego_states) - 1):
            # Current positions and velocities
            ego_pos = np.array([ego_states[i]['x'], ego_states[i]['y']])
            other_pos = np.array([other_states[i]['x'], other_states[i]['y']])
            
            ego_vel = ego_states[i].get('vx', 0)
            other_vel = other_states[i].get('vx', 0)
            
            # Relative velocity and position
            rel_pos = other_pos - ego_pos
            rel_vel = other_vel - ego_vel
            
            # DRAC calculation
            distance = np.linalg.norm(rel_pos)
            if rel_vel > 0 and distance > 0:
                drac = (rel_vel ** 2) / (2 * distance)
                dracs.append(drac)
        
        return max(dracs) if dracs else 0.0
    
    def compute_pet(self, states, agent1, agent2):
        """Post Encroachment Time (PET)."""
        # Find when agents are closest
        min_distance = float('inf')
        closest_time = 0
        
        for state in states:
            if agent1 in state.agents and agent2 in state.agents:
                pos1 = np.array([state.agents[agent1]['x'], state.agents[agent1]['y']])
                pos2 = np.array([state.agents[agent2]['x'], state.agents[agent2]['y']])
                
                distance = np.linalg.norm(pos2 - pos1)
                if distance < min_distance:
                    min_distance = distance
                    closest_time = state.time
        
        # PET is time difference when agents occupy same space
        # Simplified: return time margin at closest approach
        return closest_time
    
    def compute_comfort_metrics(self, states, agent_id):
        """Compute comfort metrics (jerk, lateral acceleration)."""
        agent_states = [state.agents[agent_id] for state in states if agent_id in state.agents]
        
        if len(agent_states) < 3:
            return {'jerk': 0, 'lateral_accel': 0}
        
        # Calculate derivatives
        positions = np.array([[s['x'], s['y']] for s in agent_states])
        times = np.array([state.time for state in states if agent_id in state.agents])
        
        # Velocity (first derivative)
        velocities = np.gradient(positions, times, axis=0)
        
        # Acceleration (second derivative)
        accelerations = np.gradient(velocities, times, axis=0)
        
        # Jerk (third derivative)
        jerks = np.gradient(accelerations, times, axis=0)
        
        # Comfort metrics
        max_jerk = np.max(np.linalg.norm(jerks, axis=1))
        max_lateral_accel = np.max(np.abs(accelerations[:, 1]))  # y-component
        
        return {
            'max_jerk': max_jerk,
            'max_lateral_acceleration': max_lateral_accel,
            'rms_jerk': np.sqrt(np.mean(np.sum(jerks**2, axis=1))),
            'comfort_score': 1.0 / (1.0 + max_jerk + max_lateral_accel)
        }

# Usage
advanced_metrics = AdvancedSafetyMetrics()

# Run simulation with advanced metrics
log = ADSimulator(scene).run(headless=True)

# Compute additional metrics
ego_states = [state.agents['ego'] for state in log.states]
ped_states = [state.agents['ped_0'] for state in log.states]

drac = advanced_metrics.compute_drac(ego_states, ped_states)
pet = advanced_metrics.compute_pet(log.states, 'ego', 'ped_0')
comfort = advanced_metrics.compute_comfort_metrics(log.states, 'ego')

print(f"DRAC: {drac:.2f} m/s²")
print(f"PET: {pet:.2f} s")
print(f"Comfort score: {comfort['comfort_score']:.3f}")
```

### statistical Analysis Framework

```python
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class StatisticalAnalysis:
    """Advanced statistical analysis for simulation results."""
    
    def __init__(self, results_df):
        self.df = results_df.copy()
        self.scaler = StandardScaler()
        
    def correlation_analysis(self, target_column='is_safe'):
        """Analyze correlations between parameters and safety outcomes."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corr()[target_column].sort_values(key=abs, ascending=False)
        
        return correlations[correlations.index != target_column]
    
    def effect_size_analysis(self, parameter_col, outcome_col='n_collisions'):
        """Compute effect sizes for parameter variations."""
        # Split data by parameter value (median split)
        median_val = self.df[parameter_col].median()
        
        low_group = self.df[self.df[parameter_col] <= median_val][outcome_col]
        high_group = self.df[self.df[parameter_col] > median_val][outcome_col]
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(low_group) - 1) * np.var(low_group) + 
                             (len(high_group) - 1) * np.var(high_group)) / 
                            (len(low_group) + len(high_group) - 2))
        
        cohens_d = (np.mean(high_group) - np.mean(low_group)) / pooled_std
        
        # Statistical significance
        t_stat, p_value = stats.ttest_ind(low_group, high_group)
        
        return {
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_magnitude': 'small' if abs(cohens_d) < 0.5 else 
                              'medium' if abs(cohens_d) < 0.8 else 'large'
        }
    
    def dimensionality_reduction(self, parameter_cols):
        """PCA analysis of parameter space."""
        X = self.df[parameter_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        return {
            'explained_variance_ratio': explained_var,
            'cumulative_variance': cumulative_var,
            'components': pca.components_,
            'transformed_data': X_pca
        }
    
    def scenario_clustering(self, feature_cols, n_clusters=5):
        """Cluster scenarios by similarity."""
        X = self.df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.df['cluster'] = clusters
        
        # Analyze cluster characteristics
        cluster_summary = self.df.groupby('cluster')[feature_cols + ['is_safe', 'n_collisions']].agg([
            'mean', 'std', 'count'
        ]).round(3)
        
        return {
            'cluster_labels': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_summary': cluster_summary
        }

# Usage example
statistical_analysis = StatisticalAnalysis(combined_df)

# Correlation analysis
correlations = statistical_analysis.correlation_analysis('n_collisions')
print("Parameter correlations with collision risk:")
print(correlations)

# Effect size for time delay
delay_effects = statistical_analysis.effect_size_analysis('delay', 'n_collisions')
print(f"Time delay effect size: {delay_effects['cohens_d']:.3f} ({delay_effects['effect_magnitude']})")

# PCA analysis
pca_results = statistical_analysis.dimensionality_reduction(['delay', 'speed_scale'])
print(f"First 2 PCs explain {pca_results['cumulative_variance'][1]:.1%} of variance")
```
---

## Integration with Planning Systems
### Gym-like Environment Interface

```python
import gym
from gym import spaces
import numpy as np

class ADSimLiteEnv(gym.Env):
    """OpenAI Gym environment wrapper for planner evaluation."""
    
    def __init__(self, scenario_file, scenario_id, dt=0.05):
        super().__init__()
        
        self.base_scene = load_scenario(scenario_file, scenario_id)
        self.dt = dt
        self.current_time = 0.0
        self.max_time = self.base_scene.duration
        
        # Action space: [acceleration, steering_angle]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -0.5]),  # max decel, max left steer
            high=np.array([3.0, 0.5]),   # max accel, max right steer
            dtype=np.float32
        )
        
        # Observation space: [ego_x, ego_y, ego_v, ped_x, ped_y, ped_v, relative_distance]
        self.observation_space = spaces.Box(
            low=np.array([-100, -10, 0, -100, -10, 0, 0]),
            high=np.array([100, 10, 30, 100, 10, 5, 50]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_scene = self.base_scene.copy()
        self.current_time = 0.0
        self.simulator = ADSimulator(self.current_scene, dt=self.dt)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one environment step."""
        # Apply action to ego vehicle (modify trajectory)
        acceleration, steering = action
        self._apply_ego_action(acceleration, steering)
        
        # Step simulation
        self.current_time += self.dt
        state = self.simulator.step(self.current_time)
        
        # Compute reward
        reward = self._compute_reward(state)
        
        # Check if episode is done
        done = (self.current_time >= self.max_time or 
                len(state.collisions) > 0)
        
        # Additional info
        info = {
            'collisions': len(state.collisions),
            'ttc': self._compute_ttc(state),
            'distance': self._compute_min_distance(state)
        }
        
        return self._get_observation(), reward, done, info
    
    def _apply_ego_action(self, acceleration, steering):
        """Apply planner action to ego vehicle."""
        ego_agent = self.current_scene.agents['ego']
        
        # Simple kinematic model - modify future waypoints
        for wp in ego_agent.trajectory.waypoints:
            if wp.t >= self.current_time:
                # Apply acceleration
                dt = wp.t - self.current_time
                wp.v = max(0, wp.v + acceleration * dt)
                
                # Apply steering (lateral displacement)
                wp.y += steering * dt
    
    def _compute_reward(self, state):
        """Compute reward signal for RL training."""
        reward = 0.0
        
        # Collision penalty
        if state.collisions:
            reward -= 100.0
        
        # TTC-based reward
        ttc = self._compute_ttc(state)
        if ttc < 3.0:
            reward -= (3.0 - ttc) * 10.0
        
        # Progress reward
        if 'ego' in state.agents:
            ego_x = state.agents['ego']['x']
            reward += ego_x * 0.1  # Forward progress
        
        # Comfort penalty (discourage harsh maneuvers)
        # This would require velocity tracking
        
        return reward
    
    def _get_observation(self):
        """Get current observation vector."""
        if self.current_time == 0:
            # Initial observation
            ego_state = self.current_scene.agents['ego'].get_state_at(0)
            ped_state = self.current_scene.agents['ped_0'].get_state_at(0)
        else:
            state = self.simulator.step(self.current_time)
            ego_state = state.agents['ego']
            ped_state = state.agents['ped_0']
        
        # Relative distance
        dx = ped_state['x'] - ego_state['x']
        dy = ped_state['y'] - ego_state['y']
        distance = np.sqrt(dx**2 + dy**2)
        
        observation = np.array([
            ego_state['x'], ego_state['y'], ego_state['v'],
            ped_state['x'], ped_state['y'], ped_state['v'],
            distance
        ], dtype=np.float32)
        
        return observation

# Usage with RL algorithms
env = ADSimLiteEnv("scenarios/pedestrian_crossing.json", "crossing_001")

# Example random policy
for episode in range(10):
    obs = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}: reward={total_reward:.1f}, collisions={info['collisions']}")
```

---

## Performance Optimization
### Vectorized Simulation

```python
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

class BatchSimulator:
    """Optimized batch simulation for large-scale studies."""
    
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or multiprocessing.cpu_count()
    
    def run_batch_parallel(self, scenarios, dt=0.05):
        """Run scenarios in parallel using multiprocessing."""
        
        def simulate_single(scenario):
            sim = ADSimulator(scenario, dt=dt)
            log = sim.run(headless=True)
            return {
                'scenario_id': scenario.id,
                'metrics': log.metrics,
                'duration': log.duration
            }
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(simulate_single, scenarios))
        
        return results
    
    def run_parameter_sweep_vectorized(self, base_scenario, parameter_grid):
        """Vectorized parameter sweep for efficiency."""
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(product(*param_values))
        
        # Create scenarios
        scenarios = []
        perturb = PerturbationEngine()
        
        for i, params in enumerate(combinations):
            scenario = base_scenario.copy()
            param_dict = dict(zip(param_names, params))
            
            # Apply perturbations based on parameter dictionary
            if 'delay' in param_dict:
                scenario = perturb.temporal_shift(scenario, "ped_0", param_dict['delay'])
            if 'speed_scale' in param_dict:
                scenario = perturb.speed_scaling(scenario, "ped_0", param_dict['speed_scale'])
            if 'lateral_offset' in param_dict:
                scenario = perturb.lateral_nudge(scenario, "ped_0", param_dict['lateral_offset'])
            
            scenario.id = f"{base_scenario.id}_sweep_{i:04d}"
            scenarios.append((scenario, param_dict))
        
        # Run simulations
        results = []
        for scenario, params in scenarios:
            sim = ADSimulator(scenario, dt=0.05)
            log = sim.run(headless=True)
            
            result = {
                'scenario_id': scenario.id,
                **params,
                **log.metrics,
                'simulation_time': log.duration
            }
            results.append(result)
        
        return pd.DataFrame(results)

# Usage
batch_sim = BatchSimulator(n_workers=8)

# Large-scale parameter sweep
parameter_grid = {
    'delay': np.linspace(-3, 3, 31),
    'speed_scale': np.linspace(0.5, 2.0, 16),
    'lateral_offset': np.linspace(-1.0, 1.0, 11)
}

print(f"Running {np.prod([len(v) for v in parameter_grid.values()])} parameter combinations...")
sweep_results = batch_sim.run_parameter_sweep_vectorized(base_scene, parameter_grid)

print(f"Completed sweep with {len(sweep_results)} simulations")
print(f"Collision rate: {(sweep_results['n_collisions'] > 0).mean():.1%}")
```

### Memory-Efficient Data Handling

```pyhton
import h5py
import json
from pathlib import Path

class EfficientDataManager:
    """Memory-efficient data storage and retrieval."""
    
    def __init__(self, output_dir="simulation_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_batch_results_hdf5(self, results, filename="batch_results.h5"):
        """Save large result sets to HDF5 format."""
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # Create groups for different data types
            metrics_group = f.create_group('metrics')
            metadata_group = f.create_group('metadata')
            
            # Convert results to structured format
            n_results = len(results)
            
            # Metrics arrays
            for metric_name in results[0]['metrics'].keys():
                metric_data = [r['metrics'][metric_name] for r in results]
                metrics_group.create_dataset(metric_name, data=metric_data)
            
            # Metadata
            scenario_ids = [r['scenario_id'].encode('utf-8') for r in results]
            metadata_group.create_dataset('scenario_ids', data=scenario_ids)
            
            # Attributes
            f.attrs['n_scenarios'] = n_results
            f.attrs['created_at'] = str(pd.Timestamp.now())
    
    def load_batch_results_hdf5(self, filename="batch_results.h5"):
        """Load results from HDF5 format."""
        filepath = self.output_dir / filename
        
        results = []
        with h5py.File(filepath, 'r') as f:
            n_scenarios = f.attrs['n_scenarios']
            scenario_ids = [sid.decode('utf-8') for sid in f['metadata']['scenario_ids']]
            
            for i in range(n_scenarios):
                result = {
                    'scenario_id': scenario_ids[i],
                    'metrics': {}
                }
                
                for metric_name in f['metrics'].keys():
                    result['metrics'][metric_name] = f['metrics'][metric_name][i]
                
                results.append(result)
        
        return results
```
