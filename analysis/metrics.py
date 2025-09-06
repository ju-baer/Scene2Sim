"""
Analysis and evaluation metrics.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json

from ..core.scene import Scene, SceneObject
from ..core.simulator import SimulationResult, SimulationFrame

@dataclass
class SceneMetrics:
    """Comprehensive scene analysis metrics."""
    object_count: int
    object_diversity: float  # Shannon entropy of object types
    spatial_distribution: float  # Spatial variance
    scene_complexity: float  # Overall complexity score
    detection_confidence: float  # Average detection confidence
    depth_range: Tuple[float, float]  # Min/max depth
    coverage_ratio: float  # Fraction of image covered by objects

@dataclass
class SimulationMetrics:
    """Simulation quality and performance metrics."""
    total_collisions: int
    collision_rate: float  # Collisions per second
    average_velocity: float
    max_velocity: float
    object_persistence: float  # How long objects remain in scene
    simulation_stability: float  # Measure of numerical stability
    frame_rate: float  # Actual simulation frame rate

class MetricsCalculator:
    """Calculate various analysis and simulation metrics."""
    
    def __init__(self):
        pass
    
    def calculate_scene_metrics(self, scene: Scene, image: Optional[np.ndarray] = None) -> SceneMetrics:
        """Calculate comprehensive scene metrics."""
        objects = list(scene.objects.values())
        
        if not objects:
            return SceneMetrics(
                object_count=0,
                object_diversity=0.0,
                spatial_distribution=0.0,
                scene_complexity=0.0,
                detection_confidence=0.0,
                depth_range=(0.0, 0.0),
                coverage_ratio=0.0
            )
        
        # Object count and diversity
        object_count = len(objects)
        type_counts = {}
        confidences = []
        positions = []
        depths = []
        
        for obj in objects:
            obj_type = obj.object_type.value
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            confidences.append(obj.confidence)
            positions.append((obj.position.x, obj.position.y, obj.position.z))
            depths.append(abs(obj.position.z))
        
        # Calculate diversity (Shannon entropy)
        total_objects = sum(type_counts.values())
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total_objects
                entropy -= p * np.log2(p)
        
        # Spatial distribution (variance of positions)
        if len(positions) > 1:
            positions_array = np.array(positions)
            spatial_var = np.mean(np.var(positions_array, axis=0))
        else:
            spatial_var = 0.0
        
        # Scene complexity (combination of factors)
        complexity = (object_count / 10.0) * entropy * (spatial_var / 100.0)
        
        # Detection confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Depth range
        depth_range = (min(depths), max(depths)) if depths else (0.0, 0.0)
        
        # Coverage ratio (requires image)
        coverage_ratio = 0.0
        if image is not None:
            coverage_ratio = self._calculate_coverage_ratio(objects, image)
        
        return SceneMetrics(
            object_count=object_count,
            object_diversity=entropy,
            spatial_distribution=spatial_var,
            scene_complexity=complexity,
            detection_confidence=avg_confidence,
            depth_range=depth_range,
            coverage_ratio=coverage_ratio
        )
    
    def _calculate_coverage_ratio(self, objects: List[SceneObject], image: np.ndarray) -> float:
        """Calculate what fraction of the image is covered by detected objects."""
        h, w = image.shape[:2]
        total_pixels = h * w
        covered_pixels = 0
        
        # Create mask of covered areas
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for obj in objects:
            if '2d_bbox' in obj.metadata:
                x, y, bbox_w, bbox_h = obj.metadata['2d_bbox']
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x + bbox_w), min(h, y + bbox_h)
                mask[y:y2, x:x2] = 1
        
        covered_pixels = np.sum(mask)
        return covered_pixels / total_pixels
    
    def calculate_simulation_metrics(self, result: SimulationResult) -> SimulationMetrics:
        """Calculate simulation performance metrics."""
        frames = result.frames
        
        if not frames:
            return SimulationMetrics(
                total_collisions=0,
                collision_rate=0.0,
                average_velocity=0.0,
                max_velocity=0.0,
                object_persistence=0.0,
                simulation_stability=1.0,
                frame_rate=0.0
            )
        
        # Collision metrics
        total_collisions = sum(len(frame.collisions) for frame in frames)
        simulation_time = result.total_time
        collision_rate = total_collisions / simulation_time if simulation_time > 0 else 0.0
        
        # Velocity metrics
        all_velocities = []
        for frame in frames:
            for obj_data in frame.objects.values():
                vel = obj_data.get('velocity', [0, 0, 0])
                speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                all_velocities.append(speed)
        
        avg_velocity = np.mean(all_velocities) if all_velocities else 0.0
        max_velocity = max(all_velocities) if all_velocities else 0.0
        
        # Object persistence (how long objects stay in scene)
        object_lifetimes = {}
        for frame in frames:
            for obj_id in frame.objects.keys():
                if obj_id not in object_lifetimes:
                    object_lifetimes[obj_id] = [frame.time, frame.time]
                else:
                    object_lifetimes[obj_id][1] = frame.time
        
        lifetimes = [end - start for start, end in object_lifetimes.values()]
        avg_persistence = np.mean()
