"""
3D visualization and web rendering utilities.
"""
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.scene import Scene, SceneObject, Vector3D, ObjectType
from ..core.simulator import SimulationFrame, SimulationResult

class ThreeJSExporter:
    """Export scenes to Three.js compatible format."""
    
    def __init__(self):
        self.geometry_cache = {}
        self.material_cache = {}
    
    def export_scene(self, scene: Scene, output_path: str):
        """Export scene to Three.js JSON format."""
        scene_data = self._scene_to_threejs(scene)
        
        output_path = Path(output_path)
        output_path.write_text(json.dumps(scene_data, indent=2))
    
    def export_animation(self, simulation_result: SimulationResult, output_path: str):
        """Export simulation as Three.js animation."""
        animation_data = self._simulation_to_threejs(simulation_result)
        
        output_path = Path(output_path)
        output_path.write_text(json.dumps(animation_data, indent=2))
    
    def _scene_to_threejs(self, scene: Scene) -> Dict[str, Any]:
        """Convert Scene to Three.js format."""
        return {
            "metadata": {
                "version": 4.5,
                "type": "Object",
                "generator": "Scene2Sim"
            },
            "scene": {
                "uuid": f"scene-{scene.id}",
                "type": "Scene",
                "name": scene.id,
                "background": self._color_to_hex(scene.environment.background_color),
                "children": self._export_objects(scene.objects)
            },
            "camera": self._export_camera(scene.camera),
            "lights": self._export_lights(scene.environment),
            "materials": self._export_materials(scene.objects),
            "geometries": self._export_geometries(scene.objects)
        }
    
    def _simulation_to_threejs(self, result: SimulationResult) -> Dict[str, Any]:
        """Convert simulation result to Three.js animation format."""
        return {
            "metadata": {
                "version": 4.5,
                "type": "Animation", 
                "generator": "Scene2Sim"
            },
            "scene_id": result.scene_id,
            "duration": result.total_time,
            "fps": 30,  # Target FPS for animation
            "tracks": self._export_animation_tracks(result.frames)
        }
    
    def _export_objects(self, objects: Dict[str, SceneObject]) -> List[Dict[str, Any]]:
        """Export objects to Three.js format."""
        threejs_objects = []
        
        for obj in objects.values():
            threejs_obj = {
                "uuid": f"object-{obj.id}",
                "type": "Mesh",
                "name": obj.id,
                "position": [obj.position.x, obj.position.y, obj.position.z],
                "rotation": [obj.rotation.x, obj.rotation.y, obj.rotation.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
                "geometry": f"geometry-{obj.object_type.value}",
                "material": f"material-{obj.id}",
                "userData": {
                    "objectType": obj.object_type.value,
                    "confidence": obj.confidence,
                    "metadata": obj.metadata
                }
            }
            threejs_objects.append(threejs_obj)
        
        return threejs_objects
    
    def _export_camera(self, camera) -> Dict[str, Any]:
        """Export camera to Three.js format.""" 
        return {
            "uuid": "main-camera",
            "type": "PerspectiveCamera",
            "fov": camera.fov,
            "aspect": camera.aspect_ratio,
            "near": camera.near_plane,
            "far": camera.far_plane,
            "position": [camera.position.x, camera.position.y, camera.position.z],
            "target": [camera.target.x, camera.target.y, camera.target.z]
        }
    
    def _export_lights(self, environment) -> List[Dict[str, Any]]:
        """Export lighting to Three.js format."""
        return [
            {
                "uuid": "directional-light",
                "type": "DirectionalLight",
                "color": self._color_to_hex(environment.lighting_color),
                "intensity": environment.lighting_intensity,
                "position": [
                    environment.lighting_direction.x * 10,
                    environment.lighting_direction.y * 10, 
                    environment.lighting_direction.z * 10
                ]
            },
            {
                "uuid": "ambient-light",
                "type": "AmbientLight",
                "color": self._color_to_hex(environment.ambient_color),
                "intensity": 0.4
            }
        ]
    
    def _export_materials(self, objects: Dict[str, SceneObject]) -> List[Dict[str, Any]]:
        """Export materials to Three.js format."""
        materials = []
        
        for obj in objects.values():
            material = {
                "uuid": f"material-{obj.id}",
                "type": "MeshStandardMaterial",
                "name": f"{obj.id}-material",
                "color": self._color_to_hex(obj.material.color),
                "roughness": obj.material.roughness,
                "metalness": obj.material.metallic,
                "transparent": obj.material.transparency > 0,
                "opacity": 1.0 - obj.material.transparency
            }
            materials.append(material)
        
        return materials
    
    def _export_geometries(self, objects: Dict[str, SceneObject]) -> List[Dict[str, Any]]:
        """Export geometries to Three.js format."""
        geometries = []
        geometry_types = set()
        
        # Collect unique geometry types
        for obj in objects.values():
            geometry_types.add(obj.object_type)
        
        # Create geometries for each type
        for obj_type in geometry_types:
            geometry = self._create_geometry_for_type(obj_type)
            geometries.append(geometry)
        
        return geometries
    
    def _create_geometry_for_type(self, obj_type: ObjectType) -> Dict[str, Any]:
        """Create Three.js geometry for object type."""
        if obj_type == ObjectType.PERSON:
            # Capsule-like geometry (cylinder + spheres)
            return {
                "uuid": f"geometry-{obj_type.value}",
                "type": "CapsuleGeometry", 
                "radius": 0.3,
                "length": 1.7
            }
        elif obj_type == ObjectType.VEHICLE:
            # Box geometry
            return {
                "uuid": f"geometry-{obj_type.value}",
                "type": "BoxGeometry",
                "width": 4.5,
                "height": 1.5,
                "depth": 2.0
            }
        elif obj_type == ObjectType.BUILDING:
            # Large box
            return {
                "uuid": f"geometry-{obj_type.value}",
                "type": "BoxGeometry", 
                "width": 10,
                "height": 15,
                "depth": 10
            }
        else:
            # Default box
            return {
                "uuid": f"geometry-{obj_type.value}",
                "type": "BoxGeometry",
                "width": 1,
                "height": 1, 
                "depth": 1
            }
    
    def _export_animation_tracks(self, frames: List[SimulationFrame]) -> List[Dict[str, Any]]:
        """Export animation tracks for objects."""
        tracks = []
        
        # Collect all object IDs
        all_object_ids = set()
        for frame in frames:
            all_object_ids.update(frame.objects.keys())
        
        # Create tracks for each object
        for obj_id in all_object_ids:
            # Position track
            times = []
            positions = []
            
            for frame in frames:
                if obj_id in frame.objects:
                    times.append(frame.time)
                    pos = frame.objects[obj_id]['position']
                    positions.extend(pos)
            
            if times:
                tracks.append({
                    "name": f"object-{obj_id}.position",
                    "type": "VectorKeyframeTrack",
                    "times": times,
                    "values": positions
                })
                
                # Rotation track (if available)
                rotations = []
                for frame in frames:
                    if obj_id in frame.objects:
                        rot = frame.objects[obj_id].get('rotation', [0, 0, 0])
                        rotations.extend(rot)
                
                if rotations:
                    tracks.append({
                        "name": f"object-{obj_id}.rotation",
                        "type": "VectorKeyframeTrack", 
                        "times": times,
                        "values": rotations
                    })
        
        return tracks
    
    def _color_to_hex(self, color: Tuple[float, float, float]) -> int:
        """Convert RGB color to hex integer."""
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        return (r << 16) | (g << 8) | b
    
    def generate_html_viewer(self, scene_file: str, animation_file: Optional[str] = None) -> str:
        """Generate HTML viewer for Three.js scene."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Scene2Sim 3D Viewer</title>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; font-family: Arial; }
        #container { width: 100vw; height: 100vh; }
        #controls { position: absolute; top: 10px; left: 10px; z-index: 100; }
        button { margin: 5px; padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        #info { position: absolute; top: 10px; right: 10px; color: white; background: rgba(0,0,0,0.7); padding: 10px; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="container"></div>
    <div id="controls">
        <button onclick="resetCamera()">Reset Camera</button>
        <button onclick="toggleAnimation()">Play/Pause</button>
        <button onclick="toggleWireframe()">Wireframe</button>
    </div>
    <div id="info">
        <div>Scene: Loading...</div>
        <div>Objects: 0</div>
        <div>Time: 0.00s</div>
    </div>
    
    <script>
        let scene, camera, renderer, controls;
        let objects = {};
        let animationMixer = null;
        let isPlaying = false;
        
        init();
        animate();
        
        function init() {
            // Scene setup
            scene = new THREE.Scene();
            
            // Camera setup
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 10, 20);
            
            // Renderer setup  
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Load scene
            loadScene('""" + scene_file + """');
            
            // Load animation if available
            """ + (f"loadAnimation('{animation_file}');" if animation_file else "") + """
            
            // Window resize handler
            window.addEventListener('resize', onWindowResize);
        }
        
        function loadScene(sceneFile) {
            fetch(sceneFile)
                .then(response => response.json())
                .then(data => {
                    createSceneFromData(data);
                })
                .catch(error => console.error('Error loading scene:', error)); }
            function createSceneFromData(data) {
        // Set background
        if (data.scene.background) {
            scene.background = new THREE.Color(data.scene.background);
        }
        
        // Create materials
        const materials = {};
        data.materials.forEach(matData => {
            const material = new THREE.MeshStandardMaterial({
                color: matData.color,
                roughness: matData.roughness,
                metalness: matData.metalness,
                transparent: matData.transparent,
                opacity: matData.opacity
            });
            materials[matData.uuid] = material;
        });
        
        // Create geometries
        const geometries = {};
        data.geometries.forEach(geomData => {
            let geometry;
            switch (geomData.type) {
                case 'BoxGeometry':
                    geometry = new THREE.BoxGeometry(geomData.width, geomData.height, geomData.depth);
                    break;
                case 'SphereGeometry':
                    geometry = new THREE.SphereGeometry(geomData.radius || 1);
                    break;
                case 'CylinderGeometry':
                    geometry = new THREE.CylinderGeometry(
                        geomData.radiusTop || geomData.radius || 1,
                        geomData.radiusBottom || geomData.radius || 1,
                        geomData.height || 2
                    );
                    break;
                default:
                    geometry = new THREE.BoxGeometry(1, 1, 1);
            }
            geometries[geomData.uuid] = geometry;
        });
        
        // Create objects
        data.scene.children.forEach(objData => {
            const geometry = geometries[objData.geometry];
            const material = materials[objData.material];
            
            if (geometry && material) {
                const mesh = new THREE.Mesh(geometry, material);
                mesh.name = objData.name;
                mesh.position.set(...objData.position);
                mesh.rotation.set(...objData.rotation);
                mesh.scale.set(...objData.scale);
                mesh.userData = objData.userData;
                
                scene.add(mesh);
                objects[objData.name] = mesh;
            }
        });
        
        // Add lights
        data.lights.forEach(lightData => {
            let light;
            switch (lightData.type) {
                case 'DirectionalLight':
                    light = new THREE.DirectionalLight(lightData.color, lightData.intensity);
                    light.position.set(...lightData.position);
                    light.castShadow = true;
                    break;
                case 'AmbientLight':
                    light = new THREE.AmbientLight(lightData.color, lightData.intensity);
                    break;
            }
            if (light) scene.add(light);
        });
        
        // Update info
        updateInfo();
    }
    
    function loadAnimation(animationFile) {
        fetch(animationFile)
            .then(response => response.json())
            .then(data => {
                createAnimationFromData(data);
            })
            .catch(error => console.error('Error loading animation:', error));
    }
    
    function createAnimationFromData(data) {
        if (!data.tracks || data.tracks.length === 0) return;
        
        const tracks = [];
        
        data.tracks.forEach(trackData => {
            const target = trackData.name.split('.')[0];
            const property = trackData.name.split('.')[1];
            
            const times = new Float32Array(trackData.times);
            const values = new Float32Array(trackData.values);
            
            let track;
            if (property === 'position') {
                track = new THREE.VectorKeyframeTrack(trackData.name, times, values);
            } else if (property === 'rotation') {
                track = new THREE.VectorKeyframeTrack(trackData.name, times, values);
            } else if (property === 'scale') {
                track = new THREE.VectorKeyframeTrack(trackData.name, times, values);
            }
            
            if (track) tracks.push(track);
        });
        
        if (tracks.length > 0) {
            const clip = new THREE.AnimationClip('simulation', data.duration, tracks);
            animationMixer = new THREE.AnimationMixer(scene);
            const action = animationMixer.clipAction(clip);
            action.play();
        }
    }
    
    function animate() {
        requestAnimationFrame(animate);
        
        if (animationMixer && isPlaying) {
            animationMixer.update(0.016); // ~60 FPS
        }
        
        renderer.render(scene, camera);
    }
    
    function resetCamera() {
        camera.position.set(0, 10, 20);
        camera.lookAt(0, 0, 0);
    }
    
    function toggleAnimation() {
        if (animationMixer) {
            isPlaying = !isPlaying;
            if (isPlaying) {
                animationMixer.timeScale = 1;
            } else {
                animationMixer.timeScale = 0;
            }
        }
    }
    
    function toggleWireframe() {
        Object.values(objects).forEach(obj => {
            if (obj.material) {
                obj.material.wireframe = !obj.material.wireframe;
            }
        });
    }
    
    function updateInfo() {
        const infoDiv = document.getElementById('info');
        infoDiv.innerHTML = `
            <div>Objects: ${Object.keys(objects).length}</div>
            <div>Animation: ${animationMixer ? 'Loaded' : 'None'}</div>
            <div>Playing: ${isPlaying ? 'Yes' : 'No'}</div>
        `;
    }
    
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    // Mouse controls
    let mouseDown = false, mouseX = 0, mouseY = 0;
    
    document.addEventListener('mousedown', (event) => {
        mouseDown = true;
        mouseX = event.clientX;
        mouseY = event.clientY;
    });
    
    document.addEventListener('mouseup', () => {
        mouseDown = false;
    });
    
    document.addEventListener('mousemove', (event) => {
        if (!mouseDown) return;
        
        const deltaX = event.clientX - mouseX;
        const deltaY = event.clientY - mouseY;
        
        // Rotate camera around scene
        const spherical = new THREE.Spherical();
        spherical.setFromVector3(camera.position);
        spherical.theta -= deltaX * 0.01;
        spherical.phi += deltaY * 0.01;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
        
        camera.position.setFromSpherical(spherical);
        camera.lookAt(0, 0, 0);
        
        mouseX = event.clientX;
        mouseY = event.clientY;
    });
    
    // Zoom with mouse wheel
    document.addEventListener('wheel', (event) => {
        const zoomSpeed = 0.1;
        const direction = camera.position.clone().normalize();
        
        if (event.deltaY > 0) {
            camera.position.add(direction.multiplyScalar(-zoomSpeed));
        } else {
            camera.position.add(direction.multiplyScalar(zoomSpeed));
        }
    });
</script>
