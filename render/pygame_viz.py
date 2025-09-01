"""
Real-time 2D visualization using Pygame.
"""
import pygame
import numpy as np
from typing import Dict, Tuple, Optional
import sys

from ..core.scene import Scene, AgentType
from ..core.simulator import SimulationState

class PygameRenderer:
    """Interactive 2D renderer for scenarios."""
    
    def __init__(self, scene: Scene, width: int = 1200, height: int = 800,
                 fps: int = 20, scale: float = 10.0):
        """
        Initialize renderer.
        
        Args:
            scene: Scenario to visualize
            width, height: Window dimensions
            fps: Target frame rate
            scale: Pixels per meter
        """
        pygame.init()
        
        self.scene = scene
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps
        
        # Initialize display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"AD-SimLite: {scene.id}")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Colors
        self.colors = {
            'background': (40, 40, 40),
            'road': (60, 60, 60),
            'lane_line': (200, 200, 200),
            'ego': (0, 255, 0),
            'pedestrian': (255, 100, 100),
            'vehicle': (100, 100, 255),
            'collision': (255, 255, 0),
            'text': (255, 255, 255)
        }
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # View parameters
        self.camera_x = 0
        self.camera_y = 0
        
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen pixels."""
        screen_x = int((x - self.camera_x) * self.scale + self.width // 2)
        screen_y = int(self.height // 2 - (y - self.camera_y) * self.scale)
        return screen_x, screen_y
    
    def draw_road(self):
        """Draw road surface and lane markings."""
        road = self.scene.road
        
        # Road surface
        road_top_y = road.width / 2
        road_bottom_y = -road.width / 2
        
        top_screen = self.world_to_screen(-100, road_top_y)[1]
        bottom_screen = self.world_to_screen(-100, road_bottom_y)[1]
        
        pygame.draw.rect(self.screen, self.colors['road'],
                        (0, top_screen, self.width, bottom_screen - top_screen))
        
        # Lane lines
        for i in range(road.n_ego_lanes + road.n_opposite_lanes + 1):
            lane_y = road_bottom_y + i * road.lane_width
            y_screen = self.world_to_screen(0, lane_y)[1]
            
            pygame.draw.line(self.screen, self.colors['lane_line'],
                           (0, y_screen), (self.width, y_screen), 2)
        
        # Center line (dashed)
        center_y_screen = self.world_to_screen(0, 0)[1]
        for x in range(0, self.width, 20):
            if (x // 20) % 2 == 0:
                pygame.draw.line(self.screen, self.colors['lane_line'],
                               (x, center_y_screen), (x + 10, center_y_screen), 3)
    
    def draw_agent(self, agent_id: str, state: Dict[str, float],
                   is_collision: bool = False):
        """Draw single agent."""
        agent = self.scene.agents[agent_id]
        
        # Position and size
        x, y = state['x'], state['y']
        length, width = state['length'], state['width']
        
        # Choose color
        if is_collision:
            color = self.colors['collision']
        elif agent.agent_type == AgentType.EGO:
            color = self.colors['ego']
        elif agent.agent_type == AgentType.PEDESTRIAN:
            color = self.colors['pedestrian']
        else:
            color = self.colors['vehicle']
        
        # Convert to screen coordinates
        screen_x, screen_y = self.world_to_screen(x, y)
        screen_length = int(length * self.scale)
        screen_width = int(width * self.scale)
        
        # Draw agent as rectangle
        rect = pygame.Rect(
            screen_x - screen_length // 2,
            screen_y - screen_width // 2,
            screen_length,
            screen_width
        )
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
        
        # Draw agent ID
        text = self.small_font.render(agent_id, True, self.colors['text'])
        text_rect = text.get_rect(center=(screen_x, screen_y))
        self.screen.blit(text, text_rect)
    
    def draw_trajectories(self):
        """Draw future trajectories as lines."""
        for agent_id, agent in self.scene.agents.items():
            if agent.agent_type == AgentType.EGO:
                color = (0, 150, 0)
            else:
                color = (150, 50, 50)
            
            # Draw waypoints
            points = []
            for wp in agent.trajectory.waypoints:
                screen_x, screen_y = self.world_to_screen(wp.x, wp.y)
                points.append((screen_x, screen_y))
                
                # Draw waypoint marker
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), 3)
            
            # Draw trajectory line
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
    
    def draw_hud(self, state: SimulationState, metrics: Optional[Dict] = None):
        """Draw heads-up display with info."""
        y_offset = 10
        
        # Time
        time_text = f"Time: {state.time:.2f}s"
        text_surface = self.font.render(time_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, y_offset))
        y_offset += 30
        
        # Agent states
        for agent_id, agent_state in state.agents.items():
            speed = agent_state.get('v', 0)
            info_text = f"{agent_id}: x={agent_state['x']:.1f}, y={agent_state['y']:.1f}, v={speed:.1f}"
            text_surface = self.small_font.render(info_text, True, self.colors['text'])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 20
        
        # Collisions
        if state.collisions:
            collision_text = f"COLLISIONS: {len(state.collisions)}"
            text_surface = self.font.render(collision_text, True, self.colors['collision'])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 30
        
        # Controls
        controls = ["ESC: Quit", "SPACE: Pause", "Arrow Keys: Pan Camera"]
        for control in controls:
            text_surface = self.small_font.render(control, True, (180, 180, 180))
            self.screen.blit(text_surface, (10, self.height - 60 + controls.index(control) * 15))
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
        
        # Camera controls
        keys = pygame.key.get_pressed()
        camera_speed = 2.0 / self.scale  # world units per frame
        
        if keys[pygame.K_LEFT]:
            self.camera_x -= camera_speed
        if keys[pygame.K_RIGHT]:
            self.camera_x += camera_speed
        if keys[pygame.K_UP]:
            self.camera_y += camera_speed
        if keys[pygame.K_DOWN]:
            self.camera_y -= camera_speed
    
    def render(self, state: SimulationState, metrics: Optional[Dict] = None):
        """Render single frame."""
        self.handle_events()
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Update camera to follow ego
        ego_agent = self.scene.get_ego_agent()
        if ego_agent and 'ego' in state.agents:
            ego_state = state.agents['ego']
            self.camera_x = ego_state['x'] - 10  # Offset behind ego
        
        # Draw elements
        self.draw_road()
        self.draw_trajectories()
        
        # Draw agents
        collision_agents = set()
        for pair in state.collisions:
            collision_agents.update(pair)
        
        for agent_id, agent_state in state.agents.items():
            is_collision = agent_id in collision_agents
            self.draw_agent(agent_id, agent_state, is_collision)
        
        # Draw HUD
        self.draw_hud(state, metrics)
        
        # Update display
        pygame.display.flip()
    
    def quit(self):
        """Clean shutdown."""
        pygame.quit()
