# Dataset Integration Guide

Scene2Sim supports multiple annotation formats for seamless integration with existing datasets.

## Supported Formats

| Format | Status | Features |
|--------|--------|----------|
| **JSON** | Native | Full feature support |
| **CVAT** | Native | Trajectory import |
| **nuScenes** | Beta | Scene conversion |
| **Waymo** | Beta | Trajectory extraction |
| **KITTI** | Planned | Object tracking |
| **Cityscapes** | Planned | Scene understanding |

## JSON Format (Native)

### Schema Overview
```json
{
  "id": "scenario_001",
  "roadConfiguration": {
    "roadWidth": 14.0,
    "nEgoDirectionLanes": 2,
    "nEgoOppositeDirectionLanes": 2
  },
  "egoConfiguration": {
    "egoLaneWrtCenter": 0,
    "egoSpeedStart": 13.89,
    "egoSpeedEnd": 13.89
  },
  "path": [
    {
      "location": {
        "laneId": -1,
        "laneSection": "RIGHT",
        "distanceToInitialEgo": -5.0
      },
      "behavior": {
        "speed": 1.4
      }
    }
  ]
}
```

### Loading JSON Data
```python
from Scene2Sim.io.json_adapter import JSONAdapter

# Single scenario
scene = JSONAdapter.load_from_file("dataset.json", "scenario_001")

# Batch loading
scenarios = []
scenario_ids = ["scenario_001", "scenario_002", "scenario_003"]
for sid in scenario_ids:
    scene = JSONAdapter.load_from_file("dataset.json", sid)
    scenarios.append(scene)
```

## CVAT Integration

### Export from CVAT
1. Open your CVAT project
2. Go to **Actions** → **Export task dataset**
3. Select **CVAT for video 1.1** format
4. Download and extract

### Loading CVAT Data
```python
from Scene2Sim.io.cvat_adapter import CVATAdapter

# Load CVAT annotations
scene = CVATAdapter.load_from_directory("cvat_export/", scene_id="task_1")

# Configure coordinate system
scene.road.configure_coordinates(
    origin=(0, 0),
    scale=0.1,  # pixels to meters
    rotation=0  # degrees
)
```

## nuScenes Integration

### Prerequisites
```bash
pip install nuscenes-devkit
```

### Loading nuScenes Data
```python
from Scene2Sim.io.nuscenes_adapter import NuScenesAdapter

# Initialize nuScenes dataset
adapter = NuScenesAdapter("/path/to/nuscenes", version="v1.0-mini")

# Convert scene
scene = adapter.convert_scene("scene-0001", duration=10.0)

# Filter by agent types
scene = adapter.convert_scene(
    "scene-0001", 
    duration=10.0,
    include_agents=['vehicle.car', 'human.pedestrian']
)
```

## Custom Dataset Integration

### Creating Custom Adapter
```python
from Scene2Sim.io.base_adapter import BaseAdapter
from Scene2Sim.core.scene import Scene, Agent, Trajectory, Waypoint

class MyDatasetAdapter(BaseAdapter):
    
    def load_from_file(self, filepath: str, scene_id: str) -> Scene:
        # Load your data
        data = self.load_my_format(filepath, scene_id)
        
        # Create road configuration
        road = self.parse_road_config(data['road'])
        
        # Create scene
        scene = Scene(id=scene_id, road=road)
        
        # Add agents
        for agent_data in data['agents']:
            agent = self.parse_agent(agent_data)
            scene.add_agent(agent)
        
        return scene
    
    def parse_agent(self, agent_data) -> Agent:
        # Convert your format to AD-SimLite format
        waypoints = []
        for wp_data in agent_data['trajectory']:
            wp = Waypoint(
                t=wp_data['timestamp'],
                x=wp_data['position'][0],
                y=wp_data['position'][1],
                v=wp_data['velocity']
            )
            waypoints.append(wp)
        
        trajectory = Trajectory(waypoints=waypoints)
        
        return Agent(
            id=agent_data['id'],
            agent_type=AgentType.PEDESTRIAN,  # or appropriate type
            trajectory=trajectory,
            length=agent_data['dimensions']['length'],
            width=agent_data['dimensions']['width']
        )
```

## Data Validation

### Automatic Validation
```python
from Scene2Sim.io.validators import SceneValidator

validator = SceneValidator()
issues = validator.validate(scene)

for issue in issues:
    print(f"{issue.severity}: {issue.message}")
    if issue.fix_suggestion:
        print(f"  Suggestion: {issue.fix_suggestion}")
```

### Manual Validation
```python
# Check coordinate system
print(f"Road width: {scene.road.width}m")
print(f"Lane count: ego={scene.road.n_ego_lanes}, opp={scene.road.n_opposite_lanes}")

# Check agent trajectories
for agent_id, agent in scene.agents.items():
    traj = agent.trajectory
    print(f"{agent_id}: {len(traj.waypoints)} waypoints, duration={traj.waypoints[-1].t:.1f}s")
    
    # Check for issues
    if len(traj.waypoints) < 2:
        print(f"  WARNING: {agent_id} has only {len(traj.waypoints)} waypoints")
```

## Export Options

### OpenSCENARIO Export
```python
from Scene2Sim.io.exporters import OpenSCENARIOExporter

exporter = OpenSCENARIOExporter()
exporter.export_scene(scene, "scenario_001.xosc")
```

### ROS2 Integration
```python
from Scene2Sim.io.ros2_bridge import ROS2Publisher

# Publish simulation state to ROS2
publisher = ROS2Publisher(topic="/simulation/state")

sim = ADSimulator(scene)
for state in sim.run_generator():
    publisher.publish_state(state)
```

## Best Practices

### Data Preprocessing
1. **Coordinate System**: Ensure consistent coordinate frames
2. **Units**: Convert all measurements to meters/seconds
3. **Timing**: Verify trajectory timestamps are monotonic
4. **Validation**: Always run validator before simulation

### Quality Assurance
```python
# Check data quality
quality_report = validator.assess_quality(scene)
print(f"Data quality score: {quality_report.score}/100")
print(f"Issues: {len(quality_report.issues)}")

# Automatic fixes
if quality_report.auto_fixable:
    fixed_scene = validator.apply_fixes(scene, quality_report.fixes)
```

### Performance Tips
- **Batch Loading**: Use generators for large datasets
- **Memory Management**: Process scenes individually for large datasets
- **Caching**: Cache processed scenes to avoid re-conversion

---
