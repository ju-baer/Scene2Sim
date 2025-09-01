"""
Geometric utilities for collision detection and transforms.
"""
import numpy as np
from typing import Tuple

def check_collision(agent1: Tuple[float, float, float, float],
                   agent2: Tuple[float, float, float, float]) -> bool:
    """
    Check collision between two rectangular agents.
    
    Args:
        agent1: (x, y, length, width) 
        agent2: (x, y, length, width)
        
    Returns:
        True if agents are colliding
    """
    x1, y1, l1, w1 = agent1
    x2, y2, l2, w2 = agent2
    
    # Simple AABB (axis-aligned bounding box) collision
    # For MVP, assume agents are axis-aligned rectangles
    
    # Agent 1 bounds
    left1 = x1 - l1/2
    right1 = x1 + l1/2  
    top1 = y1 + w1/2
    bottom1 = y1 - w1/2
    
    # Agent 2 bounds
    left2 = x2 - l2/2
    right2 = x2 + l2/2
    top2 = y2 + w2/2
    bottom2 = y2 - w2/2
    
    # Check overlap
    x_overlap = not (right1 < left2 or right2 < left1)
    y_overlap = not (top1 < bottom2 or top2 < bottom1)
    
    return x_overlap and y_overlap

def point_in_rectangle(point: Tuple[float, float],
                      rect_center: Tuple[float, float],
                      rect_size: Tuple[float, float]) -> bool:
    """Check if point is inside axis-aligned rectangle."""
    px, py = point
    cx, cy = rect_center
    length, width = rect_size
    
    return (abs(px - cx) <= length/2 and 
            abs(py - cy) <= width/2)

def compute_distance(pos1: Tuple[float, float],
                    pos2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    x1, y1 = pos1
    x2, y2 = pos2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
