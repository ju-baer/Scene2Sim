"""
Core functionality tests.
"""
import pytest
import numpy as np
from scene2sim.core.scene import Scene, SceneObject, Vector3D, ObjectType
from scene2sim.core.simulator import Simulator
from scene2sim.core.perturbations import PerturbationEngine

def test_vector3d_operations():
    """Test Vector3D mathematical operations."""
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    
    # Addition
    v3 = v1 + v2
    assert v3.x == 5 and v3.y == 7 and v3.z == 9
    
    # Subtraction
    v4 = v2 - v1
    assert v4.x == 3 and v4.y == 3 and v4.z == 3
    
    # Scalar multiplication
    v5 = v1 * 2
    assert v5.x == 2 and v5.y == 4 and v5.z == 6
    
    # Magnitude
    v6 = Vector3D(3, 4, 0)
    assert v6.magnitude() == 5.0
    
    # Normalization
    v7 = v6.normalize()
    assert abs(v7.magnitude() - 1.0) < 1e-6

def test_scene_creation():
    """Test scene creation and object management."""
    scene = Scene(scene_id="test")
    
    # Add objects
    obj1 = SceneObject(
        id="obj1",
        object_type=ObjectType.VEHICLE,
        position=Vector3D(0, 0, 0)
    )
    scene.add_object(obj1)
    
    assert len(scene.objects) == 1
    assert scene.get_object("obj1") == obj1
    
    # Remove object
    removed = scene.remove_object("obj1")
    assert removed == obj1
    assert len(scene.objects) == 0

def test_simulation_basic():
    """Test basic simulation functionality."""
    scene = Scene(scene_id="sim_test")
    
    # Add moving object
    obj = SceneObject(
        id="moving",
        object_type=ObjectType.VEHICLE,
        position=Vector3D(0, 0, 0),
        velocity=Vector3D(1, 0, 0)
    )
    scene.add_object(obj)
    
    # Run simulation
    simulator = Simulator(scene, enable_physics=False)
    result = simulator.run(duration=2.0)
    
    assert len(result.frames) > 0
    assert result.total_time >= 2.0
    
    # Check object moved
    first_frame = result.frames[0]
    last_frame = result.frames[-1]
    
    first_pos = first_frame.objects["moving"]["position"]
    last_pos = last_frame.objects["moving"]["position"]
    
    assert last_pos[0] > first_pos[0]  # Object moved in x direction

def test_perturbations():
    """Test scene perturbation system."""
    scene = Scene(scene_id="perturb_test")
    
    obj = SceneObject(
        id="test_obj",
        object_type=ObjectType.PERSON,
        position=Vector3D(0, 0, 0)
    )
    scene.add_object(obj)
    
    engine = PerturbationEngine()
    
    # Test translation
    translated = engine.translate_object(scene, "test_obj", Vector3D(1, 2, 3))
    new_pos = translated.get_object("test_obj").position
    
    assert new_pos.x == 1 and new_pos.y == 2 and new_pos.z == 3
    
    # Test scaling
    scaled = engine.scale_object(scene, "test_obj", 2.0)
    new_scale = scaled.get_object("test_obj").scale
    
    assert new_scale.x == 2.0 and new_scale.y == 2.0 and new_scale.z == 2.0

def test_scene_serialization():
    """Test scene saving and loading."""
    import tempfile
    import os
    
    # Create scene
    scene = Scene(scene_id="serialize_test")
    obj = SceneObject(
        id="test",
        object_type=ObjectType.BUILDING,
        position=Vector3D(5, 10, 15),
        confidence=0.95
    )
    scene.add_object(obj)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        scene.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Load from file
        loaded_scene = Scene.load(tmp_path)
        
        assert loaded_scene.id == scene.id
        assert len(loaded_scene.objects) == 1
        
        loaded_obj = loaded_scene.get_object("test")
        assert loaded_obj.object_type == ObjectType.BUILDING
        assert loaded_obj.position.x == 5
        assert loaded_obj.confidence == 0.95
        
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    pytest.main([__file__])
