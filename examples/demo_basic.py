"""
Basic Scene2Sim demonstration script.
"""
import sys
from pathlib import Path

# Add scene2sim to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scene2sim import Scene, SceneObject, Vector3D, ObjectType, Simulator, load_image
from scene2sim.core.perturbations import PerturbationEngine
from scene2sim.analysis.metrics import MetricsCalculator

def main():
    print("Scene2Sim Basic Demo")
    print("=" * 50)
    
    # Create a simple scene manually
    scene = Scene(scene_id="demo_basic")
    
    # Add ground plane
    ground = SceneObject(
        id="ground",
        object_type=ObjectType.GROUND,
        position=Vector3D(0, -1, 0),
        scale=Vector3D(20, 0.1, 20)
    )
    ground.physics.is_static = True
    scene.add_object(ground)
    
    # Add a moving vehicle
    vehicle = SceneObject(
        id="car1",
        object_type=ObjectType.VEHICLE,
        position=Vector3D(-5, 0, 0),
        velocity=Vector3D(2, 0, 0),
        scale=Vector3D(4, 1.5, 2)
    )
    scene.add_object(vehicle)
    
    # Add a person
    person = SceneObject(
        id="person1",
        object_type=ObjectType.PERSON,
        position=Vector3D(3, 0, 2),
        velocity=Vector3D(-0.5, 0, -0.3),
        scale=Vector3D(0.6, 1.8, 0.6)
    )
    scene.add_object(person)
    
    print(f"Created scene with {len(scene.objects)} objects")
    
    # Calculate scene metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_scene_metrics(scene)
    
    print(f"\nScene Metrics:")
    print(f"  Objects: {metrics.object_count}")
    print(f"  Diversity: {metrics.object_diversity:.2f}")
    print(f"  Complexity: {metrics.scene_complexity:.2f}")
    
    # Run simulation
    print(f"\nRunning simulation...")
    simulator = Simulator(scene, enable_physics=True, enable_collisions=True)
    
    result = simulator.run(duration=5.0)
    
    print(f"Simulation completed:")
    print(f"  Duration: {result.total_time:.2f}s")
    print(f"  Frames: {len(result.frames)}")
    print(f"  Wall time: {result.wall_clock_time:.2f}s")
    print(f"  Collisions: {result.final_metrics.get('total_collisions', 0)}")
    
    # Test perturbations
    print(f"\nTesting perturbations...")
    perturb_engine = PerturbationEngine()
    
    # Create variants
    variants = perturb_engine.create_scenario_variants(scene, n_variants=3)
    
    for i, variant in enumerate(variants):
        print(f"  Variant {i+1}: {len(variant.objects)} objects")
    
    # Save scene
    output_path = Path("demo_scene.json")
    scene.save(str(output_path))
    print(f"\nScene saved to: {output_path}")
    
    # Test loading
    loaded_scene = Scene.load(str(output_path))
    print(f"Loaded scene: {loaded_scene.id} with {len(loaded_scene.objects)} objects")
    
    print(f"\nDemo completed successfully!")

if __name__ == "__main__":
    main()
