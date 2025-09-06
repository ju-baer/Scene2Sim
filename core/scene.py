"""
Core scene representation and data structures.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path

class ObjectType(Enum):
    """Types of objects that can exist in a scene."""
    PERSON = "person"
    VEHICLE = "vehicle"
    BUILDING = "building"
    FURNITURE = "furniture"
    VEGETATION = "vegetation"
    GROUND = "ground"
    SKY = "sky"
    UNKNOWN = "unknown"

class MotionType(Enum):
    """Types of motion patterns."""
    STATIC = "static"
    LINEAR = "linear"
    CIRCULAR = "circular"
    RANDOM = "random"
    PHYSICS = "physics"

@dataclass
class Vector3D:
    """3D vector representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D()
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)

@dataclass
class BoundingBox:
    """3D bounding box representation."""
    center: Vector3D
    size: Vector3D  # width, height, depth
    rotation: Vector3D = field(default_factory=Vector3D)  # euler angles
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if point is inside bounding box."""
        # Simplified AABB check (ignoring rotation for now)
        return (abs(point.x - self.center.x) <= self.size.x/2 and
                abs(point.y - self.center.y) <= self.size.y/2 and
                abs(point.z - self.center.z) <= self.size.z/2)

@dataclass
class Material:
    """Material properties for objects."""
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)  # RGB
    roughness: float = 0.5
    metallic: float = 0.0
    transparency: float = 0.0
    texture_path: Optional[str] = None

@dataclass
class Physics:
    """Physics properties for simulation."""
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.3  # bounciness
    is_static: bool = False
    gravity_affected: bool = True

@dataclass
class SceneObject:
    """Individual object in the scene."""
    id: str
    object_type: ObjectType
    position: Vector3D
    rotation: Vector3D = field(default_factory=Vector3D)
    scale: Vector3D = field(default_factory=lambda: Vector3D(1.0, 1.0, 1.0))
    bounding_box: Optional[BoundingBox] = None
    material: Material = field(default_factory=Material)
    physics: Physics = field(default_factory=Physics)
    
    # Motion properties
    velocity: Vector3D = field(default_factory=Vector3D)
    angular_velocity: Vector3D = field(default_factory=Vector3D)
    motion_type: MotionType = MotionType.STATIC
    
    # Additional data
    confidence: float = 1.0  # detection confidence
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_position(self, dt: float):
        """Update object position based on velocity."""
        if self.motion_type == MotionType.STATIC:
            return
            
        if self.motion_type == MotionType.LINEAR:
            self.position = self.position + (self.velocity * dt)
            self.rotation = self.rotation + (self.angular_velocity * dt)
        elif self.motion_type == MotionType.PHYSICS:
            # Will be handled by physics engine
            pass

@dataclass
class Camera:
    """Camera configuration for the scene."""
    position: Vector3D = field(default_factory=lambda: Vector3D(0, 0, 5))
    target: Vector3D = field(default_factory=Vector3D)
    up: Vector3D = field(default_factory=lambda: Vector3D(0, 1, 0))
    
    # Camera parameters
    fov: float = 45.0  # field of view in degrees
    aspect_ratio: float = 16.0/9.0
    near_plane: float = 0.1
    far_plane: float = 1000.0
    
    # Intrinsic parameters (if known)
    focal_length: Optional[float] = None
    sensor_size: Optional[Tuple[float, float]] = None

@dataclass
class Environment:
    """Environmental settings for the scene."""
    lighting_direction: Vector3D = field(default_factory=lambda: Vector3D(1, -1, 1))
    lighting_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    lighting_intensity: float = 1.0
    ambient_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    background_color: Tuple[float, float, float] = (0.5, 0.7, 1.0)
    fog_enabled: bool = False
    fog_density: float = 0.01

class Scene:
    """Main scene container with all objects and metadata."""
    
    def __init__(self, scene_id: str = "default", source_path: Optional[str] = None):
        self.id = scene_id
        self.source_path = source_path
        self.objects: Dict[str, SceneObject] = {}
        self.camera = Camera()
        self.environment = Environment()
        
        # Scene metadata
        self.timestamp = None
        self.duration = 0.0  # for video scenes
        self.frame_rate = 30.0
        self.resolution = (1920, 1080)
        
        # Analysis results
        self.analysis_complete = False
        self.analysis_metadata: Dict[str, Any] = {}
    
    def add_object(self, obj: SceneObject) -> None:
        """Add object to scene."""
        self.objects[obj.id] = obj
    
    def remove_object(self, object_id: str) -> Optional[SceneObject]:
        """Remove object from scene."""
        return self.objects.pop(object_id, None)
    
    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Get object by ID."""
        return self.objects.get(object_id)
    
    def get_objects_by_type(self, object_type: ObjectType) -> List[SceneObject]:
        """Get all objects of specific type."""
        return [obj for obj in self.objects.values() 
                if obj.object_type == object_type]
    
    def get_bounds(self) -> Optional[BoundingBox]:
        """Get overall scene bounding box."""
        if not self.objects:
            return None
            
        positions = [obj.position for obj in self.objects.values()]
        
        min_x = min(pos.x for pos in positions)
        max_x = max(pos.x for pos in positions)
        min_y = min(pos.y for pos in positions)  
        max_y = max(pos.y for pos in positions)
        min_z = min(pos.z for pos in positions)
        max_z = max(pos.z for pos in positions)
        
        center = Vector3D(
            (min_x + max_x) / 2,
            (min_y + max_y) / 2, 
            (min_z + max_z) / 2
        )
        
        size = Vector3D(
            max_x - min_x,
            max_y - min_y,
            max_z - min_z
        )
        
        return BoundingBox(center=center, size=size)
    
    def update(self, dt: float) -> None:
        """Update all objects in scene."""
        for obj in self.objects.values():
            obj.update_position(dt)
    
    def copy(self) -> 'Scene':
        """Create deep copy of scene."""
        import copy
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene to dictionary for serialization."""
        return {
            'id': self.id,
            'source_path': self.source_path,
            'objects': {obj_id: {
                'id': obj.id,
                'type': obj.object_type.value,
                'position': [obj.position.x, obj.position.y, obj.position.z],
                'rotation': [obj.rotation.x, obj.rotation.y, obj.rotation.z],
                'scale': [obj.scale.x, obj.scale.y, obj.scale.z],
                'velocity': [obj.velocity.x, obj.velocity.y, obj.velocity.z],
                'motion_type': obj.motion_type.value,
                'material': {
                    'color': obj.material.color,
                    'roughness': obj.material.roughness,
                    'metallic': obj.material.metallic,
                    'transparency': obj.material.transparency
                },
                'physics': {
                    'mass': obj.physics.mass,
                    'friction': obj.physics.friction,
                    'restitution': obj.physics.restitution,
                    'is_static': obj.physics.is_static
                },
                'confidence': obj.confidence,
                'metadata': obj.metadata
            } for obj_id, obj in self.objects.items()},
            'camera': {
                'position': [self.camera.position.x, self.camera.position.y, self.camera.position.z],
                'target': [self.camera.target.x, self.camera.target.y, self.camera.target.z],
                'fov': self.camera.fov,
                'aspect_ratio': self.camera.aspect_ratio
            },
            'environment': {
                'lighting_direction': [self.environment.lighting_direction.x, 
                                     self.environment.lighting_direction.y,
                                     self.environment.lighting_direction.z],
                'lighting_intensity': self.environment.lighting_intensity,
                'background_color': self.environment.background_color
            },
            'metadata': {
                'duration': self.duration,
                'frame_rate': self.frame_rate,
                'resolution': self.resolution,
                'analysis_complete': self.analysis_complete,
                'analysis_metadata': self.analysis_metadata
            }
        }
    
    def save(self, filepath: str) -> None:
        """Save scene to JSON file."""
        Path(filepath).write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, filepath: str) -> 'Scene':
        """Load scene from JSON file."""
        data = json.loads(Path(filepath).read_text())
        
        scene = cls(data['id'], data.get('source_path'))
        
        # Load objects
        for obj_data in data['objects'].values():
            obj = SceneObject(
                id=obj_data['id'],
                object_type=ObjectType(obj_data['type']),
                position=Vector3D(*obj_data['position']),
                rotation=Vector3D(*obj_data['rotation']),
                scale=Vector3D(*obj_data['scale']),
                velocity=Vector3D(*obj_data['velocity']),
                motion_type=MotionType(obj_data['motion_type']),
                confidence=obj_data['confidence'],
                metadata=obj_data['metadata']
            )
            
            # Load material
            mat_data = obj_data['material']
            obj.material = Material(
                color=tuple(mat_data['color']),
                roughness=mat_data['roughness'],
                metallic=mat_data['metallic'],
                transparency=mat_data['transparency']
            )
            
            # Load physics
            phys_data = obj_data['physics']
            obj.physics = Physics(
                mass=phys_data['mass'],
                friction=phys_data['friction'],
                restitution=phys_data['restitution'],
                is_static=phys_data['is_static']
            )
            
            scene.add_object(obj)
        
        # Load camera
        cam_data = data['camera']
        scene.camera.position = Vector3D(*cam_data['position'])
        scene.camera.target = Vector3D(*cam_data['target'])
        scene.camera.fov = cam_data['fov']
        scene.camera.aspect_ratio = cam_data['aspect_ratio']
        
        # Load environment
        env_data = data['environment']
        scene.environment.lighting_direction = Vector3D(*env_data['lighting_direction'])
        scene.environment.lighting_intensity = env_data['lighting_intensity']
        scene.environment.background_color = tuple(env_data['background_color'])
        
        # Load metadata
        meta_data = data['metadata']
        scene.duration = meta_data['duration']
        scene.frame_rate = meta_data['frame_rate']
        scene.resolution = tuple(meta_data['resolution'])
        scene.analysis_complete = meta_data['analysis_complete']
        scene.analysis_metadata = meta_data['analysis_metadata']
        
        return scene
