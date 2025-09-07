# visualisation/visualizer.py

import copy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vpython import *
from param_config import SHOW_VISUALS, FLOOR_COUNT, FLOOR_WIDTH, FLOOR_LENGTH, FLOOR_HEIGHT, STAIR_HEIGHT, STAIR_WIDTH, STAIR_DEPTH,DOOR_WIDTH, STAIR_CAPACITY, STAIR_GAP, FLOOR_OPACITY, WALL_OPACITY, STAIR_OPACITY, DOOR_OPACITY
import threading
from vpython import box, sphere, cylinder, vector, color, scene, rate, compound, canvas
import os

class VisualizerVPython:
    def __init__(self, env, num_humans=20, mover_type="random", algorithm_name="random"):
        if not SHOW_VISUALS:
            return  # Do not initialize any VPython scene or objects if visuals are off
        self.env = env
        self.num_humans = num_humans
        self.scene = scene
        self.vector = vector
        self.color = color
        self.box = box
        self.sphere = sphere
        self.cylinder = cylinder
        self.compound = compound
        self.rate = rate
        self.stick_figures = []
        self._running = False
        self.stick_template = None
        self.mover_type=mover_type
        self.algorithm_name = algorithm_name

        canvas_id = f'canvas_{self.algorithm_name.upper()}'
        self.graph = graph(width=400, height=250, align='right',
                          title=f'{self.algorithm_name.upper()} Evacuation Progress',
                          xtitle='Time (s)', ytitle='Evacuated')
        
        self.unique_colors = set()

    def _setup_scene(self):
        if not SHOW_VISUALS:
            return
        try:
            # Cache all common values to reduce property lookups
            floor_width = getattr(self.env, "width", 30)
            floor_length = getattr(self.env, "length", 30)
            floor_thickness = getattr(self.env, "floor_thickness", 0.5)
            floor_height = getattr(self.env, "floor_height", 10) if hasattr(self.env, "floor_height") else 10
            num_floors = getattr(self.env, "floor_count", 3)
            
            # Pre-compute vectors and colors once
            floor_color = self.vector(0.7, 0.7, 0.9)
            wall_color = self.vector(0.8, 0.8, 0.8)
            stair_color = self.vector(0.7, 0.6, 0.8)
            exit_color = self.vector(0.9, 0.5, 0.5)
            interior_door_color = self.vector(1.0, 0.6, 0.1)
            
            # Cache common values
            opacity_floor = FLOOR_OPACITY
            opacity_wall = WALL_OPACITY
            opacity_stair = STAIR_OPACITY
            room_wall_thickness = 0.3
            door_height = 3
            door_width = DOOR_WIDTH
            door_thickness = 0.1
            
            # Pre-compute common offsets
            half_width = floor_width/2
            half_length = floor_length/2

            # Set up scene once
            if self.mover_type.lower() == "random":
                self.scene.title = f"{self.algorithm_name} Pedestrian Simulation"
                self.scene.width = 800
                self.scene.height = 600
                self.scene.autoscale = True
                self.scene.background = self.color.white
                self.scene.center = self.vector(0, (num_floors-1)*floor_height/2, 0)
                self.scene.forward = self.vector(-1, -0.5, -1)
            else:
                self.scene = canvas(title=f"{self.algorithm_name} Pedestrian Simulation",
                           width=800, height=600, autoscale=True,
                           background=self.color.white)
                self.scene.center = self.vector(0, (num_floors-1)*floor_height/2, 0)
                self.scene.forward = self.vector(-1, -0.5, -1)

            # Create all floors at once as compound object
            floor_objects = []
            floor_size = self.vector(floor_width, floor_thickness, floor_length)
            for i in range(num_floors):
                floor_objects.append(
                    self.box(pos=self.vector(0, i*floor_height, 0),
                            size=floor_size,
                            color=floor_color,
                            opacity=opacity_floor,
                            make_trail=False)
                )
            self.floors = floor_objects
            # Create all doors (exits and interior) at once
            door_objects = []
            
            # Exit doors
            for e in self.env.exits:
                door_objects.append(
                    self.box(
                        pos=self.vector(
                            e['x']-half_width,
                            door_height/2 + floor_thickness/2,
                            e['y']-half_length
                        ),
                        size=self.vector(door_width, door_height, door_thickness),
                        color=exit_color,
                        opacity=DOOR_OPACITY,
                        make_trail=False
                    )
                )
            
            # Interior doors
            if hasattr(self.env, "get_doors"):
                for (dx, dy, direction) in self.env.get_doors():
                    for i in range(num_floors):
                        door_pos = self.vector(
                            dx - half_width,
                            door_height/2 + i*floor_height + floor_thickness/2,
                            dy - half_length
                        )
                        door_size = (self.vector(door_thickness, door_height, door_width) 
                                   if direction == "vertical" 
                                   else self.vector(door_width, door_height, door_thickness))
                        door_objects.append(
                            self.box(
                                pos=door_pos,
                                size=door_size,
                                color=interior_door_color,
                                opacity=DOOR_OPACITY,
                                make_trail=False
                            )
                        )
            
            self.exits = []
            self.interior_doors = door_objects
            # Create all walls at once
            wall_height = 3
            wall_objects = []
            for (x1, y1, x2, y2) in self.env.get_walls():
                for i in range(num_floors):
                    y = i*floor_height + floor_thickness/2 + wall_height/2
                    # Vertical wall
                    wall_objects.append(
                        self.box(
                            pos=self.vector((x1+x2)/2-half_width, y, y1-half_length),
                            size=self.vector(abs(x2-x1), wall_height, room_wall_thickness),
                            color=wall_color,
                            opacity=opacity_wall,
                            make_trail=False
                        )
                    )
                    # Horizontal wall
                    wall_objects.append(
                        self.box(
                            pos=self.vector(x1-half_width, y, (y1+y2)/2-half_length),
                            size=self.vector(room_wall_thickness, wall_height, abs(y2-y1)),
                            color=wall_color,
                            opacity=opacity_wall,
                            make_trail=False
                        )
                    )
            self.inner_walls = wall_objects

            # Create all stairs at once
            stair_objects = []
            stair_size = self.vector(STAIR_WIDTH, STAIR_HEIGHT, STAIR_DEPTH)
            for s in self.env.staircases:
                if s.start == (5, 25):
                    start_x, start_y = 5, 25
                    end_x, end_y = 5, 5
                else:
                    start_x, start_y = 25, 25
                    end_x, end_y = 25, 5
                    
                z0 = s.from_floor * (floor_height + floor_thickness)
                z1 = s.to_floor * (floor_height + floor_thickness)
                num_steps = int(abs(z1 - z0) / STAIR_HEIGHT)
                
                dx = (end_x - start_x) / num_steps
                dy = (end_y - start_y) / num_steps
                dz = (z1 - z0) / num_steps
                
                for i in range(num_steps):
                    x = start_x + dx * i
                    y = start_y + dy * i
                    z = z0 + dz * i
                    stair_objects.append(
                        self.box(
                            pos=self.vector(x - half_width, z, y - half_length),
                            size=stair_size,
                            color=stair_color,
                            opacity=opacity_stair,
                            make_trail=False
                        )
                    )
            self.staircases = stair_objects
            # Stick figures
            self.stick_figures = []
            
            stick_colors = [self.color.blue]
            # Stick human height (should match pedestrian height)
            stick_foot_offset = 0.5  # feet at y=0
            for i in range(self.num_humans):
                stick = self._make_stick_human(stick_colors[i % len(stick_colors)])
                # Place feet on the first floor, using floor_height and floor_thickness

                #Get floor number of pedestrian
                floor_num=0
                y_pos = floor_num * floor_height + floor_thickness / 2 + stick_foot_offset + 1.75
                stick.pos = self.vector(0, y_pos, 0)
                stick.floor_num = floor_num
                stick.visible = True
                self.stick_figures.append(stick)
        except Exception as e:
            print(f"[DEBUG][VISUALIZER] Error in _setup_scene: {e}")

    def _make_stick_human(self, color_val):
        """Create a grouped stick figure with color support."""
        # Fixed dimensions for all stick figures
        dims = {
            'head': (0.12, 0.12),  # radius, height
            'body': (0.11, 1.1),   # radius, height
            'leg': (0.07, 0.5),    # radius, length
            'arm': (0.06, 0.45)    # radius, length
        }
        
        class StickGroup:
            def __init__(self, parts, color, vector_fn):
                self.parts = parts
                self._color = color
                self._pos = vector_fn(0, 0, 0)
                self.axis = vector_fn(1, 0, 0)
                self.up = vector_fn(0, 1, 0)
                self.visible = True
                self.vector = vector_fn  # Store vector function
                
                # Store initial relative positions
                for part in parts:
                    part._original_pos = part.pos - self._pos
            
            @property
            def color(self):
                return self._color
                
            @color.setter
            def color(self, value):
                self._color = value
                for part in self.parts:
                    part.color = value
            
            @property
            def pos(self):
                return self._pos
                
            @pos.setter
            def pos(self, new_pos):
                self._pos = new_pos
                for part in self.parts:
                    part.pos = new_pos + part._original_pos
            
            @property
            def visible(self):
                return self._visible
                
            @visible.setter
            def visible(self, value):
                self._visible = value
                for part in self.parts:
                    part.visible = value
                    
            def rotate(self, angle=0, axis=None):
                for part in self.parts:
                    part.rotate(angle=angle, axis=axis, origin=self._pos)
        
        # Create individual parts
        parts = []
        
        # Head
        parts.append(self.sphere(
            pos=self.vector(0, dims['leg'][1] + dims['body'][1] + dims['head'][0], 0),
            radius=dims['head'][0],
            color=color_val
        ))
        
        # Body
        parts.append(self.cylinder(
            pos=self.vector(0, dims['leg'][1], 0),
            axis=self.vector(0, dims['body'][1], 0),
            radius=dims['body'][0],
            color=color_val
        ))
        
        # Arms
        parts.append(self.cylinder(
            pos=self.vector(-0.22, dims['leg'][1] + dims['body'][1] - 0.15, 0),
            axis=self.vector(-dims['arm'][1], -0.18, 0),
            radius=dims['arm'][0],
            color=color_val
        ))
        
        parts.append(self.cylinder(
            pos=self.vector(0.22, dims['leg'][1] + dims['body'][1] - 0.15, 0),
            axis=self.vector(dims['arm'][1], -0.18, 0),
            radius=dims['arm'][0],
            color=color_val
        ))
        
        # Legs
        parts.append(self.cylinder(
            pos=self.vector(-0.09, 0, 0),
            axis=self.vector(0, dims['leg'][1], 0),
            radius=dims['leg'][0],
            color=color_val
        ))
        
        parts.append(self.cylinder(
            pos=self.vector(0.09, 0, 0),
            axis=self.vector(0, dims['leg'][1], 0),
            radius=dims['leg'][0],
            color=color_val
        ))
        
        # Create and return the group
        return StickGroup(parts, color_val, self.vector)
    

    def update(self, pedestrian_list, evac_time=None):
        if not SHOW_VISUALS:
            return
        stick_foot_offset = 0  # feet at y=0
        color_map = {
            'yellow': self.color.yellow,
            'green': self.color.green,
            'red': self.color.red,
            'blue': self.color.blue
        }
        for i, ped in enumerate(pedestrian_list):
            try:
                if i >= len(self.stick_figures):
                    break
                stick = self.stick_figures[i]
                if getattr(ped, 'evacuated', False):
                    stick.visible = False
                    continue
                stick.visible = True
                stick.pos = self.vector(
                    ped.x - self.env.width/2,
                    ped.z + stick_foot_offset,
                    ped.y - self.env.width/2
                )
                stick.floor_num = ped.floor
                # --- Set stick color based on ped.color ---
                c = ped.color
                if c != 'blue':
                    self.unique_colors.add(c.strip().lower())
                    
                new_color = color_map.get(c.strip().lower(), self.color.blue)
                if stick.color != new_color:
                    stick.color = new_color
                # --- End stick color logic ---
                # --- Rotate stick figure to face movement direction ---
                direction = None
                if hasattr(ped,'on_stairs'):
                    direction = np.array([0,0])
                if hasattr(ped, 'velocity') and isinstance(ped.velocity, (np.ndarray, list, tuple)):
                    v = np.array(ped.velocity)
                    if np.linalg.norm(v) > 0.01:
                        direction = v
                if direction is None:
                    if hasattr(ped, 'desired_direction'):
                        d = np.array(ped.desired_direction)
                        if np.linalg.norm(d) > 0.01:
                            direction = d
                if direction is not None:
                    angle = np.arctan2(direction[1], direction[0])
                    try:
                        stick.axis = self.vector(1, 0, 0)
                        stick.up = self.vector(0, 1, 0)
                        stick.rotate(angle=angle, axis=self.vector(0, 1, 0))
                    except Exception as e:
                        print(f"[DEBUG][VISUALIZER] Error rotating stick figure {i}: {e}")
            except Exception as e:
                print(f"[DEBUG][VISUALIZER] Error updating stick figure {i}: {e}")


    def run_live(self, get_pedestrians_fn, interval=0.05):
        if not SHOW_VISUALS:
            return
        self._running = True
        def loop():
            while self._running:
                self.rate(1/interval)
                peds = get_pedestrians_fn()
                self.update(peds)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._running = False

    
        