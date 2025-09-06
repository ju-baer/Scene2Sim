"""
Advanced object detection and tracking systems.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrackingResult:
    """Object tracking result."""
    object_id: str
    trajectory: List[Tuple[float, float]]  # (x, y) positions over time
    velocities: List[Tuple[float, float]]  # (vx, vy) velocities
    confidences: List[float]
    bboxes: List[Tuple[int, int, int, int]]
    timestamps: List[float]

class MultiObjectTracker:
    """Multi-object tracking system."""
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.disappeared = {}  # object_id -> disappeared_count
        self.tracking_history = {}  # object_id -> TrackingResult
    
    def register(self, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int], 
                confidence: float, timestamp: float) -> str:
        """Register new object."""
        object_id = str(self.next_object_id)
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        
        # Initialize tracking history
        self.tracking_history[object_id] = TrackingResult(
            object_id=object_id,
            trajectory=[centroid],
            velocities=[(0.0, 0.0)],
            confidences=[confidence],
            bboxes=[bbox],
            timestamps=[timestamp]
        )
        
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id: str):
        """Deregister object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: List[Tuple[Tuple[float, float], Tuple[int, int, int, int], float]], 
               timestamp: float) -> Dict[str, TrackingResult]:
        """Update tracker with new detections."""
        if len(detections) == 0:
            # No detections, mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.tracking_history
        
        if len(self.objects) == 0:
            # No existing objects, register all detections
            for centroid, bbox, confidence in detections:
                self.register(centroid, bbox, confidence, timestamp)
        else:
            # Match detections to existing objects
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            detection_centroids = [det[0] for det in detections]
            
            # Compute distance matrix
            distances = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, det_centroid in enumerate(detection_centroids):
                    distances[i, j] = np.sqrt(
                        (obj_centroid[0] - det_centroid[0])**2 + 
                        (obj_centroid[1] - det_centroid[1])**2
                    )
            
            # Find minimum values and sort by distance
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if distances[row, col] > self.max_distance:
                    continue
                
                # Update object
                object_id = object_ids[row]
                centroid, bbox, confidence = detections[col]
                
                # Calculate velocity
                prev_centroid = self.objects[object_id]
                prev_timestamp = self.tracking_history[object_id].timestamps[-1]
                dt = timestamp - prev_timestamp
                
                if dt > 0:
                    vx = (centroid[0] - prev_centroid[0]) / dt
                    vy = (centroid[1] - prev_centroid[1]) / dt
                else:
                    vx, vy = 0.0, 0.0
                
                # Update tracking data
                self.objects[object_id] = centroid
                self.disappeared[object_id] = 0
                
                # Update history
                history = self.tracking_history[object_id]
                history.trajectory.append(centroid)
                history.velocities.append((vx, vy))
                history.confidences.append(confidence)
                history.bboxes.append(bbox)
                history.timestamps.append(timestamp)
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections (new objects)
            unused_col_indices = set(range(0, len(detection_centroids))) - used_col_indices
            for col in unused_col_indices:
                centroid, bbox, confidence = detections[col]
                self.register(centroid, bbox, confidence, timestamp)
            
            # Handle unmatched objects (disappeared)
            unused_row_indices = set(range(0, len(object_centroids))) - used_row_indices
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.tracking_history
    
    def get_current_objects(self) -> Dict[str, Tuple[float, float]]:
        """Get current object positions."""
        return self.objects.copy()
    
    def get_tracking_results(self) -> Dict[str, TrackingResult]:
        """Get all tracking results."""
        return self.tracking_history.copy()

class FeatureExtractor:
    """Extract features from image regions for object recognition."""
    
    def __init__(self, method: str = "hog"):
        self.method = method
        
        if method == "hog":
            self.hog = cv2.HOGDescriptor()
        elif method == "orb":
            self.orb = cv2.ORB_create()
    
    def extract_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract features from bounding box region."""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return np.array([])
        
        if self.method == "hog":
            return self._extract_hog(roi)
        elif self.method == "orb":
            return self._extract_orb(roi)
        else:
            return self._extract_simple(roi)
    
    def _extract_hog(self, roi: np.ndarray) -> np.ndarray:
        """Extract HOG features."""
        # Resize ROI to standard size
        roi_resized = cv2.resize(roi, (64, 128))
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        # Compute HOG features
        features = self.hog.compute(gray)
        return features.flatten() if features is not None else np.array([])
    
    def _extract_orb(self, roi: np.ndarray) -> np.ndarray:
        """Extract ORB features."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is not None:
            # Use bag of features approach - simplified
            return np.mean(descriptors, axis=0)
        else:
            return np.array([])
    
    def _extract_simple(self, roi: np.ndarray) -> np.ndarray:
        """Extract simple color/texture features."""
        # Color histogram
        hist_b = cv2.calcHist([roi], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [8], [0, 256])
        hist_r = cv2.calcHist([roi], [2], None, [8], [0, 256])
        
        # Texture features (simple)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        features = np.concatenate([
            hist_b.flatten(),
            hist_g.flatten(), 
            hist_r.flatten(),
            [laplacian_var]
        ])
        
        return features
