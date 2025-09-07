import random
import numpy as np
from param_config import GRID_LENGTH, GRID_WIDTH, DOOR_WIDTH

class RandomMover:
    def __init__(self, pedestrians,environment):
        self.pedestrians = pedestrians
        self.environment = environment
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_LENGTH

    def staff_nearby(self, ped, radius=3.0):
        """Return True if a staff member is within a given radius of the pedestrian."""
        for other in self.pedestrians:
            if getattr(other, 'is_staff', False) and ped.floor == other.floor:
                dist = np.linalg.norm(np.array([ped.x, ped.y]) - np.array([other.x, other.y]))
                if dist < radius:
                    return True
        return False

    def move(self, evac_time):
        """
        Moves each pedestrian randomly to an adjacent cell (up, down, left, right).
        Handles stair movement and evacuation at exits.
        """
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down
        stair_height = 0.4
        evacuated_peds = 0

        for ped in self.pedestrians:
            if getattr(ped, 'evacuated', False):
                evacuated_peds += 1
                continue

            # Handle stair movement
            if getattr(ped, 'on_stairs', False):
                stair = ped.current_stair
                floor_height = self.environment.floor_height
                num_steps = int(abs(stair.to_floor - stair.from_floor) * (floor_height / stair_height))
                step_idx = getattr(ped, 'stair_step_idx', 0) + 1

                x0, y0 = stair.start
                z1 = stair.from_floor * (floor_height + self.environment.floor_thickness / 2)
                z2 = stair.to_floor * (floor_height + self.environment.floor_thickness / 2)
                dz = z2 - z1

                stair_angle_deg = 30 if x0 > 2 and y0 > 2 else 330
                stair_length = abs(dz) / np.sin(np.deg2rad(stair_angle_deg if stair_angle_deg != 0 else 1))
                dx = 0
                dy = np.sign(dz) * np.sqrt(max(stair_length**2 - dz**2, 0))
                end_x = x0 + dx
                end_y = y0 + dy

                # Interpolate position along the stair mesh
                ped.x = x0 + step_idx * (end_x - x0) / num_steps
                ped.y = y0 + step_idx * (end_y - y0) / num_steps
                ped.z = z1 + step_idx * (z2 - z1) / num_steps
                ped.stair_step_idx = step_idx

                if step_idx >= num_steps:
                    ped.floor = stair.to_floor
                    ped.z = z2
                    ped.on_stairs = False
                    ped.current_stair = None
                    ped.stair_step_idx = 0
                continue

            # Random walk on the floor avoiding walls
            dx, dy = random.choice(moves)
            new_x = min(max(ped.x + dx, 0), self.grid_width - 1)
            new_y = min(max(ped.y + dy, 0), self.grid_height - 1)

            if not self.environment.is_blocked(ped.x,ped.y,new_x, new_y):
                ped.x = new_x
                ped.y = new_y

            ped.z = ped.floor *( self.environment.floor_height + self.environment.floor_thickness / 2)

            # Check for stairs
            if ped.floor > 0:
                stair = self.environment.get_stair_from(ped.floor, ped.x, ped.y)
                if stair and abs(ped.x - stair.start[0]) < 2 and abs(ped.y - stair.start[1]) < 2:
                    ped.on_stairs = True
                    ped.current_stair = stair
                    ped.stair_step_idx = 0

            # Check for exit on ground floor
            if ped.floor == 0:
                exit = self.environment.get_nearest_exit(ped.x, ped.y)
                if abs(ped.x - exit['x']) < DOOR_WIDTH/2 and abs(ped.y - exit['y']) < DOOR_WIDTH/2:
                    if not getattr(ped, 'evacuated', False):
                        ped.mark_evacuated(evac_time)
                        evacuated_peds += 1

        print(f"Random Mover: Evacuated {evacuated_peds} pedestrians out of {len(self.pedestrians)}")
        return evacuated_peds