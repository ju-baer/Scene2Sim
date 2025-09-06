"""
2D visualization using Pygame.
"""
import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

from ..core.scene import Scene, SceneObject, Vector3D, ObjectType
from ..core.simulator import SimulationFrame

class PygameRenderer:
    """Real-time 2D rendering using Pygame."""
    
    def __init__(self, scene: Scene, width: int = 1200, height: int = 800, fps: int = 60):
        pygame.init()
        
        self.scene = scene
        self.width = width
        self.height = height
        self.fps = fps
        self.running = True
        
        # Create display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Scene2Sim - {scene.id}")
        self.clock = pygame.time.Clock()
        
        # Rendering parameters
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 20.0  # pixels per world unit
        self.show_trajectories = True
        self.show_bboxes = True
        self.show_velocities = True
        
        # Color scheme
        self.colors = {
            ObjectType.PERSON: (255, 180, 120),      # Orange
            ObjectType.VEHICLE: (120, 120, 255),     # Blue  
            ObjectType.BUILDING: (180, 180, 180),    # Gray
            ObjectType.FURNITURE: (139, 69, 19),     # Brown
            ObjectType.VEGETATION: (34, 139, 34),    # Green
            ObjectType.GROUND: (85, 107, 47),        # Dark olive
            ObjectType.SKY: (135, 206, 235),         # Sky blue
            ObjectType.UNKNOWN: (200, 200, 200)      # Light gray
        }
        
        # Trajectory history
        self.trajectory_history: Dict[str, List[Tuple[float, float]]] = {}
        
        # UI elements
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
    
    def world_to_screen(self, world_pos: Vector3D) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((world_pos.x - self.camera_x) * self.zoom + self.width // 2)
        screen_y = int((world_pos.y - self.camera_y) * self.zoom + self.height // 2)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Vector3D:
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_pos[0] - self.width // 2) / self.zoom + self.camera_x
        world_y = (screen_pos[1] - self.height // 2) / self.zoom + self.camera_y
        return Vector3D(world_x, world_y, 0)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_t:
                    self.show_trajectories = not self.show_trajectories
                elif event.key == pygame.K_b:
                    self.show_bboxes = not self.show_bboxes
                elif event.key == pygame.K_v:
                    self.show_velocities = not self.show_velocities
                elif event.key == pygame.K_r:
                    # Reset camera
                    self.camera_x = 0
                    self.camera_y = 0
                    self.zoom = 20.0
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    self.zoom *= 1.1
                elif event.button == 5:  # Mouse wheel down
                    self.zoom /= 1.1
            
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[2]:  # Right mouse button
                    dx, dy = event.rel
                    self.camera_x -= dx / self.zoom
                    self.camera_y -= dy / self.zoom
    
    def render_object(self, obj: SceneObject):
        """Render a single object."""
        screen_pos = self.world_to_screen(obj.position)
        color = self.colors.get(obj.object_type, self.colors[ObjectType.UNKNOWN])
        
        # Calculate object size on screen
        size_x = max(3, int(obj.scale.x * self.zoom))
        size_y = max(3, int(obj.scale.y * self.zoom))
        
        # Draw object as rectangle or circle based on type
        if obj.object_type == ObjectType.PERSON:
            # Draw person as circle
            pygame.draw.circle(self.screen, color, screen_pos, max(size_x, size_y) // 2)
        else:
            # Draw as rectangle
            rect = pygame.Rect(
                screen_pos[0] - size_x // 2,
                screen_pos[1] - size_y // 2,
                size_x,
                size_y
            )
            pygame.draw.rect(self.screen, color, rect)
        
        # Draw bounding box if enabled
        if self.show_bboxes and obj.bounding_box:
            bbox_size_x = int(obj.bounding_box.size.x * self.zoom)
            bbox_size_y = int(obj.bounding_box.size.y * self.zoom)
            bbox_rect = pygame.Rect(
                screen_pos[0] - bbox_size_x // 2,
                screen_pos[1] - bbox_size_y // 2,
                bbox_size_x,
                bbox_size_y
            )
            pygame.draw.rect(self.screen, (255, 255, 255), bbox_rect, 1)
        
        # Draw velocity vector if enabled
        if self.show_velocities and obj.velocity.magnitude() > 0.1:
            vel_end = self.world_to_screen(Vector3D(
                obj.position.x + obj.velocity.x,
                obj.position.y + obj.velocity.y,
                obj.position.z
            ))
            pygame.draw.line(self.screen, (255, 255, 0), screen_pos, vel_end, 2)
            
            # Draw arrow head
            angle = math.atan2(vel_end[1] - screen_pos[1], vel_end[0] - screen_pos[0])
            arrow_len = 10
            pygame.draw.line(self.screen, (255, 255, 0), vel_end, 
                           (vel_end[0] - arrow_len * math.cos(angle - 0.3),
                            vel_end[1] - arrow_len * math.sin(angle - 0.3)), 2)
            pygame.draw.line(self.screen, (255, 255, 0), vel_end,
                           (vel_end[0] - arrow_len * math.cos(angle + 0.3),
                            vel_end[1] - arrow_len * math.sin(angle + 0.3)), 2)
        
        # Draw object ID
        if self.zoom > 10:
            text = self.small_font.render(obj.id, True, (255, 255, 255))
            text_pos = (screen_pos[0] + size_x // 2 + 5, screen_pos[1] - size_y // 2)
            self.screen.blit(text, text_pos)
    
    def render_trajectory(self, obj_id: str, positions: List[Tuple[float, float]]):
        """Render trajectory trail for an object."""
        if len(positions) < 2:
            return
        
        screen_positions = []
        for pos in positions[-50:]:  # Show last 50 positions
            world_pos = Vector3D(pos[0], pos[1], 0)
            screen_pos = self.world_to_screen(world_pos)
            screen_positions.append(screen_pos)
        
        if len(screen_positions) >= 2:
            # Draw trajectory with fading effect
            for i in range(1, len(screen_positions)):
                alpha = int(255 * (i / len(screen_positions)))
                color = (alpha, alpha // 2, alpha // 2)
                pygame.draw.line(self.screen, color, 
                               screen_positions[i-1], screen_positions[i], 2)
    
    def render_grid(self):
        """Render background grid."""
        grid_spacing = max(1, int(self.zoom))
        
        # Vertical lines
        start_x = int(-self.camera_x * self.zoom + self.width // 2) % grid_spacing
        for x in range(start_x, self.width, grid_spacing):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, self.height))
        
        # Horizontal lines  
        start_y = int(-self.camera_y * self.zoom + self.height // 2) % grid_spacing
        for y in range(start_y, self.height, grid_spacing):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (self.width, y))
        
        # Origin cross
        origin_screen = self.world_to_screen(Vector3D(0, 0, 0))
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (origin_screen[0] - 10, origin_screen[1]), 
                        (origin_screen[0] + 10, origin_screen[1]), 2)
        pygame.draw.line(self.screen, (100, 100, 100),
                        (origin_screen[0], origin_screen[1] - 10),
                        (origin_screen[0], origin_screen[1] + 10), 2)
    
    def render_ui(self, frame: Optional[SimulationFrame] = None):
        """Render UI overlay."""
        # Instructions
        instructions = [
            "Controls:",
            "ESC - Quit",
            "T - Toggle trajectories", 
            "B - Toggle bounding boxes",
            "V - Toggle velocity vectors",
            "R - Reset camera",
            "Mouse wheel - Zoom",
            "Right drag - Pan"
        ]
        
        y_offset = 10
        for instruction in instructions:
            text = self.small_font.render(instruction, True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20
        
        # Camera info
        camera_info = f"Camera: ({self.camera_x:.1f}, {self.camera_y:.1f}) Zoom: {self.zoom:.1f}x"
        camera_text = self.small_font.render(camera_info, True, (255, 255, 255))
        self.screen.blit(camera_text, (10, self.height - 60))
        
        # Frame info
        if frame:
            frame_info = f"Time: {frame.time:.2f}s Objects: {len(frame.objects)} Collisions: {len(frame.collisions)}"
            frame_text = self.small_font.render(frame_info, True, (255, 255, 255))
            self.screen.blit(frame_text, (10, self.height - 40))
            
            # Show collisions
            if frame.collisions:
                collision_text = f"COLLISION: {', '.join([f'{c[0]}-{c[1]}' for c in frame.collisions])}"
                collision_surface = self.font.render(collision_text, True, (255, 50, 50))
                self.screen.blit(collision_surface, (10, self.height - 20))
    
    def render(self, frame: Optional[SimulationFrame] = None):
        """Render complete frame."""
        # Clear screen
        self.screen.fill((20, 20, 30))  # Dark blue background
        
        # Render grid
        self.render_grid()
        
        # Update trajectory history and render trajectories
        if frame and self.show_trajectories:
            for obj_id, obj_data in frame.objects.items():
                pos = obj_data['position']
                
                if obj_id not in self.trajectory_history:
                    self.trajectory_history[obj_id] = []
                
                self.trajectory_history[obj_id].append((pos[0], pos[1]))
                self.render_trajectory(obj_id, self.trajectory_history[obj_id])
        
        # Render objects
        if frame:
            # Render from frame data
            for obj_id, obj_data in frame.objects.items():
                # Create temporary object for rendering
                temp_obj = SceneObject(
                    id=obj_id,
                    object_type=ObjectType(obj_data['type']),
                    position=Vector3D(*obj_data['position']),
                    velocity=Vector3D(*obj_data['velocity'])
                )
                temp_obj.scale = Vector3D(1, 1, 1)  # Default scale
                self.render_object(temp_obj)
        else:
            # Render from scene
            for obj in self.scene.objects.values():
                self.render_object(obj)
        
        # Render UI
        self.render_ui(frame)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()
    
    def save_screenshot(self, filepath: str):
        """Save current screen as image."""
        pygame.image.save(self.screen, filepath)
