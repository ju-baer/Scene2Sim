"""
Scene analysis and understanding using computer vision.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json

from ..core.scene import Scene, SceneObject, Vector3D, ObjectType, Material, Physics, BoundingBox

@dataclass
class Detection:
    """Object detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    mask: Optional[np.ndarray] = None
    depth: Optional[float] = None

@dataclass  
class DepthEstimate:
    """Depth estimation result."""
    depth_map: np.ndarray
    confidence_map: Optional[np.ndarray] = None
    scale_factor: float = 1.0

class ObjectDetector:
    """Object detection using various methods."""
    
    def __init__(self, method: str = "cv2", confidence_threshold: float = 0.5):
        self.method = method
        self.confidence_threshold = confidence_threshold
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the detection method."""
        if self.method == "cv2":
            # Use OpenCV's pre-trained models
            self.net = None
            self.class_names = [
                "background", "person", "bicycle", "car", "motorcycle", 
                "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                "donut", "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster",
                "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
            ]
        elif self.method == "yolo":
            # Placeholder for YOLO implementation
            pass
        elif self.method == "detectron2":
            # Placeholder for Detectron2 implementation  
            pass
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image."""
        if self.method == "cv2":
            return self._detect_cv2_simple(image)
        else:
            # Fallback to simple detection
            return self._detect_cv2_simple(image)
    
    def _detect_cv2_simple(self, image: np.ndarray) -> List[Detection]:
        """Simple object detection using OpenCV features."""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Use contour detection as simple object detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area < 100:
                continue
            
            # Get bounding box
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            
            # Skip very small or very large bboxes
            if bbox_w < 20 or bbox_h < 20 or bbox_w > w * 0.8 or bbox_h > h * 0.8:
                continue
            
            # Estimate object type based on aspect ratio and size
            aspect_ratio = bbox_w / bbox_h
            
            if aspect_ratio > 1.5 and area > 1000:
                class_name = "vehicle"
            elif aspect_ratio < 0.8 and bbox_h > bbox_w:
                class_name = "person"
            elif area > 500:
                class_name = "furniture"
            else:
                class_name = "unknown"
            
            # Simple confidence based on contour properties
            confidence = min(0.9, area / 10000.0 + 0.3)
            
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=(x, y, bbox_w, bbox_h)
            )
            detections.append(detection)
        
        return detections

class DepthEstimator:
    """Depth estimation from single images."""
    
    def __init__(self, method: str = "simple"):
        self.method = method
    
    def estimate_depth(self, image: np.ndarray) -> DepthEstimate:
        """Estimate depth map from image."""
        if self.method == "simple":
            return self._simple_depth_estimation(image)
        else:
            return self._simple_depth_estimation(image)
    
    def _simple_depth_estimation(self, image: np.ndarray) -> DepthEstimate:
        """Simple depth estimation using image features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Create depth map based on image properties
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        # Use gradient magnitude as depth cue
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and invert (high gradient = close objects)
        gradient_norm = gradient_mag / (gradient_mag.max() + 1e-6)
        
        # Use blur to estimate depth (blurred = far)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        blur_diff = np.abs(gray.astype(float) - blurred.astype(float))
        blur_norm = blur_diff / (blur_diff.max() + 1e-6)
        
        # Combine cues
        depth_map = (gradient_norm * 0.6 + blur_norm * 0.4)
        
        # Convert to actual depth values (closer = smaller values)
        depth_map = 1.0 - depth_map  # Invert
        depth_map = depth_map * 20.0 + 1.0  # Scale to 1-21 meters
        
        return DepthEstimate(depth_map=depth_map)

class SceneAnalyzer:
    """Main scene analysis pipeline."""
    
    def __init__(self, 
                 detector_method: str = "cv2",
                 depth_method: str = "simple",
                 confidence_threshold: float = 0.5):
        self.detector = ObjectDetector(detector_method, confidence_threshold)
        self.depth_estimator = DepthEstimator(depth_method)
        
        # Object type mapping
        self.type_mapping = {
            "person": ObjectType.PERSON,
            "car": ObjectType.VEHICLE,
            "truck": ObjectType.VEHICLE,
            "bus": ObjectType.VEHICLE,
            "motorcycle": ObjectType.VEHICLE,
            "bicycle": ObjectType.VEHICLE,
            "chair": ObjectType.FURNITURE,
            "couch": ObjectType.FURNITURE,
            "bed": ObjectType.FURNITURE,
            "dining table": ObjectType.FURNITURE,
            "tv": ObjectType.FURNITURE,
            "laptop": ObjectType.FURNITURE,
            "background": ObjectType.GROUND,
            "building": ObjectType.BUILDING,
        }
    
    def analyze_image(self, image: np.ndarray, scene: Scene, **kwargs) -> Scene:
        """Analyze image and populate scene with detected objects."""
        h, w = image.shape[:2]
        
        # Detect objects
        detections = self.detector.detect(image)
        
        # Estimate depth
        depth_estimate = self.depth_estimator.estimate_depth(image)
        
        # Convert detections to 3D objects
        for i, detection in enumerate(detections):
            obj = self._detection_to_3d_object(detection, depth_estimate, w, h, i)
            scene.add_object(obj)
        
        # Add ground plane
        ground_obj = SceneObject(
            id="ground",
            object_type=ObjectType.GROUND,
            position=Vector3D(0, -2, 0),
            scale=Vector3D(20, 0.1, 20)
        )
        ground_obj.physics.is_static = True
        ground_obj.material.color = (0.4, 0.6, 0.3)  # Green ground
        scene.add_object(ground_obj)
        
        # Store analysis metadata
        scene.analysis_metadata.update({
            'detections_count': len(detections),
            'detection_method': self.detector.method,
            'depth_method': self.depth_estimator.method,
            'image_resolution': (w, h),
            'objects_created': len(scene.objects)
        })
        
        return scene
    
    def _detection_to_3d_object(self, detection: Detection, depth_estimate: DepthEstimate,
                               image_width: int, image_height: int, obj_index: int) -> SceneObject:
        """Convert 2D detection to 3D scene object."""
        x, y, w, h = detection.bbox
        
        # Get average depth in bounding box region
        depth_roi = depth_estimate.depth_map[y:y+h, x:x+w]
        avg_depth = np.mean(depth_roi)
        
        # Convert 2D position to 3D world coordinates
        # Simple perspective projection (assumes known camera parameters)
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Normalize to [-1, 1] range
        norm_x = (center_x / image_width) * 2 - 1
        norm_y = -((center_y / image_height) * 2 - 1)  # Flip Y
        
        # Convert to world coordinates (simple perspective)
        fov_scale = avg_depth * 0.5  # Adjust based on field of view
        world_x = norm_x * fov_scale
        world_y = norm_y * fov_scale
        world_z = -avg_depth  # Negative Z for camera space
        
        # Estimate 3D size from 2D bounding box and depth
        scale_factor = avg_depth / 10.0  # Objects further away appear smaller
        obj_width = (w / image_width) * 10 * scale_factor
        obj_height = (h / image_height) * 10 * scale_factor
        obj_depth = min(obj_width, obj_height) * 0.5  # Estimate depth
        
        # Map detection class to object type
        obj_type = self.type_mapping.get(detection.class_name, ObjectType.UNKNOWN)
        
        # Create material based on object type
        material = Material()
        if obj_type == ObjectType.PERSON:
            material.color = (0.8, 0.6, 0.4)  # Skin tone
        elif obj_type == ObjectType.VEHICLE:
            material.color = (0.3, 0.3, 0.7)  # Blue vehicle
            material.metallic = 0.8
        elif obj_type == ObjectType.FURNITURE:
            material.color = (0.6, 0.4, 0.2)  # Brown furniture
        else:
            material.color = (0.7, 0.7, 0.7)  # Gray default
        
        # Create physics properties
        physics = Physics()
        if obj_type == ObjectType.VEHICLE:
            physics.mass = 1000.0
        elif obj_type == ObjectType.PERSON:
            physics.mass = 70.0
        elif obj_type == ObjectType.FURNITURE:
            physics.mass = 20.0
            physics.is_static = True
        
        # Create 3D object
        obj_id = f"{detection.class_name}_{obj_index}"
        obj = SceneObject(
            id=obj_id,
            object_type=obj_type,
            position=Vector3D(world_x, world_y, world_z),
            scale=Vector3D(obj_width, obj_height, obj_depth),
            material=material,
            physics=physics,
            confidence=detection.confidence
        )
        
        # Add bounding box
        obj.bounding_box = BoundingBox(
            center=obj.position,
            size=Vector3D(obj_width, obj_height, obj_depth)
        )
        
        # Store 2D detection info in metadata
        obj.metadata.update({
            '2d_bbox': detection.bbox,
            'detection_confidence': detection.confidence,
            'estimated_depth': avg_depth,
            'detection_method': self.detector.method
        })
        
        return obj
    
    def analyze_motion(self, images: List[np.ndarray], scene: Scene) -> Scene:
        """Analyze motion from sequence of images."""
        if len(images) < 2:
            return scene
        
        # Simple optical flow-based motion analysis
        prev_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(images)):
            curr_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, None, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # TODO: Associate flow vectors with detected objects
            # and update their velocities
            
            prev_gray = curr_gray
        
        # Update scene metadata
        scene.analysis_metadata['motion_analysis'] = {
            'frames_analyzed': len(images),
            'method': 'optical_flow'
        }
        
        return scene
