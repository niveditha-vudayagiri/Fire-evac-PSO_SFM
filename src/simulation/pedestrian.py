import numpy as np
import random
from param_config import DESIRED_SPEED, N_PEDESTRIANS,FLOOR_THICKNESS, FLOOR_HEIGHT

class Pedestrian:
    COLOR_PALETTE = ['blue', 'red', 'yellow', 'violet', 'maroon',
                     'magenta', 'orange', 'pink', 'brown', 'cyan']

    def __init__(self, x, y, floor, crowd_id=None, is_staff=False):
        self.x = x
        self.y = y
        self.vx = 0 
        self.vy = 0

        # For PSO
        self.px_best = x
        self.py_best = y    
        self.vx_best = 0
        self.vy_best = 0
        self.px = x
        self.py = y

        self.is_staff = is_staff
        self.following_staff = False
        self.panicked = False

        self.age = np.random.randint(10, 80)
        self.age_group = 'children' if self.age < 18 else 'adults' if self.age < 60 else 'elderly'

        # For SFM
        self.velocity = np.array([0.0, 0.0])  # 2D velocity vector
        # Set desired velocity based on age group (children, adults, elderly)
        if self.age < 18:
            self.desired_speed = np.random.uniform(1.2, 1.6)  # children: faster
        elif self.age < 60:
            self.desired_speed = np.random.uniform(1.0, 1.4)  # adults: average
        else:
            self.desired_speed = np.random.uniform(0.3, 0.8)  # elderly: slower
        self.desired_direction = np.array([0, 0, 0])
        self.acceleration = 0
        self.desired_acceleration = 0
        self.desired_acceleration = 0
        self.tau = 0.5
        self.A = 1.0
        self.B = 0.5
        self.stair_step = 0.5
        self.stair_step_idx = 0

        self.floor = floor
        self.room = None
        self.floor_thickness = FLOOR_THICKNESS  # match floor thickness in visualization
        self.z = floor * (FLOOR_HEIGHT + self.floor_thickness)  # default, will be updated by env

        self.evacuated = False
        self.evac_time = None

        # set height based on stick figure defaults
        self.height = 1.75  # average height in meters
        self.panic_level = np.random.uniform(0.5, 1.5)
        self.desired_speed = DESIRED_SPEED * self.panic_level

        self.crowd_id = crowd_id if crowd_id is not None else np.random.randint(0, N_PEDESTRIANS // 4)
        self.color = 'yellow' if is_staff else 'blue'

        self.max_speed = 1.5 * self.desired_speed
        self.on_stairs = False
        self.current_stair = None
        self.stair_progress = 0.0  # 0=start, 1=end

        self.update_color() 

    def update_color(self):
        # Staff: yellow, Following staff: green, Panicked: red, Default: blue
        # Always set color as a string for vpython/plotly mapping
        if getattr(self, 'is_staff', False):
            self.color = 'yellow'   # Staff
        elif getattr(self, 'following_staff', False):
            self.color = 'green'    # Customer following staff order
        elif getattr(self, 'panicked', False):
            self.color = 'red'      # Panicked customer not following staff
        else:
            self.color = 'blue'     # Default customer
        # Defensive: ensure color is always a string and in allowed set
        allowed = {'yellow', 'green', 'red', 'blue'}
        if self.color not in allowed:
            self.color = 'blue'

    def position(self):
        return (self.x, self.y, self.z)

    def move_on_stairs(self, stair, step_size=0.05, floor_height=10):
        """
        Move the pedestrian along the given staircase (supports both up and down).
        """
        x0, y0 = stair.start
        x1, y1 = stair.start  # default, will be set below
        if hasattr(stair, 'from_floor') and hasattr(stair, 'to_floor'):
            z0 = stair.from_floor * floor_height + self.floor_thickness
            z1 = stair.to_floor * floor_height + self.floor_thickness
            # For parallel stairs, use a fixed offset in y (z in 3D)
            if x0 < 15:
                # left stair: up in y
                x1 = x0
                y1 = y0 + 3
            else:
                # right stair: down in y
                x1 = x0
                y1 = y0 - 3
        else:
            z0 = self.z
            z1 = self.z

        # Determine direction (up or down)
        direction = 1 if stair.to_floor > stair.from_floor else -1
        self.stair_progress += step_size * direction
        # Clamp stair_progress between 0 and 1
        if direction == 1:
            if self.stair_progress > 1.0:
                self.stair_progress = 1.0
            elif self.stair_progress < 0.0:
                self.stair_progress = 0.0
        else:
            if self.stair_progress < 0.0:
                self.stair_progress = 0.0
            elif self.stair_progress > 1.0:
                self.stair_progress = 1.0

        # Interpolate position
        self.x = x0 + (x1 - x0) * self.stair_progress
        self.y = y0 + (y1 - y0) * self.stair_progress
        self.z = z0 + (z1 - z0) * self.stair_progress

        # If reached end, update floor and exit stairs
        if (direction == 1 and self.stair_progress >= 1.0) or (direction == -1 and self.stair_progress <= 0.0):
            self.floor = stair.to_floor
            self.on_stairs = False
            self.current_stair = None
            self.stair_progress = 0.0
            self.z = self.floor * (floor_height + self.floor_thickness)

    def start_stair(self, stair):
        """
        Begin moving on a staircase.
        """
        self.on_stairs = True
        self.current_stair = stair
        self.stair_progress = 0.0

    def mark_evacuated(self, evac_time):
        self.evacuated = True
        self.evac_time = evac_time

    def desired_direction(self):
        # Randomly choose a direction to move towards
        angle = random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle), 0])

    def update_position(self, floor_height=10):
        if self.on_stairs and self.current_stair is not None:
            self.move_on_stairs(self.current_stair, floor_height=floor_height)
        else:
            # Update position based on desired speed and direction
            direction = self.desired_direction()
            self.x += direction[0] * self.desired_speed
            self.y += direction[1] * self.desired_speed
            self.z = self.floor * (floor_height + self.floor_thickness)
            # Ensure the pedestrian stays within bounds
            self.x = max(0, min(self.x, 30))
            self.y = max(0, min(self.y, 30))
            self.z = max(0, min(self.z, 30))
            # Check if the pedestrian has reached the exit
            if self.x >= 30 and self.y >= 30:
                self.mark_evacuated(evac_time=0)