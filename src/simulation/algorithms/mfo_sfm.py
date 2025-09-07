import numpy as np
import random
from param_config import (GRID_WIDTH, GRID_LENGTH, TIME_STEP, PED_RADIUS, tau, mass,DOOR_WIDTH)
from .sfm_common import calculate_social_force, calculate_wall_force, process_exit_queues, handle_exit_queueing, handle_door_queueing, process_door_queues

class MFOSFMMover:
    """
    Moth-Flame Optimization inspired SFM mover.
    Each pedestrian (moth) is attracted to a "flame" (dynamic target near exits).
    """
    def __init__(self, pedestrians, environment, n_flames=5):
        self.pedestrians = pedestrians
        self.environment = environment
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_LENGTH
        self.tau = tau
        self.mass = mass
        self.n_flames = n_flames
        self.flames = self._init_flames()

    def _init_flames(self):
        # Initialize flames based on floor type
        flames = []
        for floor in range(self.environment.floor_count):
            if floor == 0:  # Ground floor: flames near exits
                for exit in self.environment.exits:
                    for _ in range(self.n_flames // len(self.environment.exits)):
                        offset = np.random.uniform(-2, 2, size=2)
                        flame_pos = np.array([exit['x'], exit['y']]) + offset
                        flame_pos[0] = np.clip(flame_pos[0], 0.5, self.grid_width - 0.5)
                        flame_pos[1] = np.clip(flame_pos[1], 0.5, self.grid_height - 0.5)
                        flames.append((floor, flame_pos))
            else:  # Upper floors: flames near stair entrances based on room
                rooms = self.environment.internal_walls + self.environment.outer_walls
                for room in rooms:
                    x1, y1, x2, y2 = room
                    stairs_in_room = [
                        stair for stair in self.environment.get_stairs_on_floor(floor)
                        if x1 <= stair.start[0] <= x2 and y1 <= stair.start[1] <= y2
                    ]
                    for stair in stairs_in_room:
                        for _ in range(self.n_flames // max(1, len(stairs_in_room))):
                            offset = np.random.uniform(-2, 2, size=2)
                            flame_pos = np.array(stair.start) + offset
                            flame_pos[0] = np.clip(flame_pos[0], 0.5, self.grid_width - 0.5)
                            flame_pos[1] = np.clip(flame_pos[1], 0.5, self.grid_height - 0.5)
                            flames.append((floor, flame_pos))
        return flames

    def _update_flames(self):
        # Optionally, flames can be updated each iteration (e.g., random near exits)
        self.flames = self._init_flames()

    def move(self, evac_time, door_queues=None):
        evacuated_peds = 0
        max_side_by_side = int(DOOR_WIDTH // (2 * PED_RADIUS))
        exit_queues = {}
        for exit in self.environment.exits:
            exit_queues[(exit['x'], exit['y'])] = []

        occupied = set()
        occupied_stairs = set()
        for ped in self.pedestrians:
            if getattr(ped, 'evacuated', False):
                continue
            if ped.on_stairs:
                occupied_stairs.add((id(ped.current_stair), ped.stair_step_idx))
            else:
                occupied.add((ped.floor, int(round(ped.x)), int(round(ped.y))))

        # Update flames each iteration (optional, or keep static for a few steps)
        self._update_flames()

        for idx, ped in enumerate(self.pedestrians):
            try:
                if getattr(ped, 'evacuated', False):
                    evacuated_peds += 1
                    continue

                # Handle stair movement
                if getattr(ped, 'on_stairs', False):
                    try:
                        from .stair_utils import move_pedestrian_on_stairs
                        move_pedestrian_on_stairs(
                            ped,
                            self.pedestrians,
                            occupied_stairs,
                            self.environment,
                            self.environment.floor_height,
                            self.environment.floor_thickness
                        )
                        continue
                    except Exception as e:
                        print(f"[DEBUG][MFO_SFM] Error moving ped on stairs: {e}")
                        continue

                pos = np.array([ped.x, ped.y])

                # Assign flame based on pedestrian's floor and room
                room = self.environment.get_room(ped.x, ped.y)
                if room:
                    x1, y1, x2, y2 = room
                    floor_flames = [
                        flame for flame_floor, flame in self.flames
                        if flame_floor == ped.floor and x1 <= flame[0] <= x2 and y1 <= flame[1] <= y2
                    ]
                else:
                    floor_flames = [
                        flame for flame_floor, flame in self.flames if flame_floor == ped.floor
                    ]

                if floor_flames:
                    flame = min(floor_flames, key=lambda f: np.linalg.norm(pos - f))
                else:
                    flame = pos  # Default to current position if no flames available

                # MFO spiral movement towards flame
                b = 1.0  # spiral shape parameter
                t = random.uniform(-1, 1)
                dist_to_flame = np.linalg.norm(flame - pos)
                spiral = dist_to_flame * np.exp(b * t) * (
                    np.cos(2 * np.pi * t) * (flame - pos) / (dist_to_flame + 1e-6)
                )
                mfo_direction = (flame - pos) + spiral
                mfo_direction_norm = np.linalg.norm(mfo_direction)
                if mfo_direction_norm > 0:
                    mfo_direction = mfo_direction / mfo_direction_norm
                else:
                    mfo_direction = np.zeros(2)

                # SFM: Social and wall forces
                if not hasattr(ped, 'velocity') or not isinstance(ped.velocity, np.ndarray):
                    ped.velocity = np.zeros(2)
                v_desired = ped.desired_speed * mfo_direction
                f_drive = (v_desired - ped.velocity) / self.tau
                from .sfm_common import amplify_drive_if_in_queue
                f_drive = amplify_drive_if_in_queue(f_drive, ped)
                f_repulsion = calculate_social_force(ped, pos, self.pedestrians)
                f_wall = calculate_wall_force(ped, pos, self.environment, self.grid_width, self.grid_height)

                force = f_drive + (f_repulsion + f_wall) / self.mass
                ped.velocity += force * TIME_STEP

                # Dampen speed if in queue
                if getattr(ped, '_in_exit_queue', False) or getattr(ped, '_in_door_queue', False):
                    ped.velocity *= 0.7  # Damping factor for queueing

                speed = np.linalg.norm(ped.velocity)
                if speed > ped.max_speed:
                    ped.velocity *= (ped.desired_speed / speed)

                new_x = ped.x + ped.velocity[0] * TIME_STEP
                new_y = ped.y + ped.velocity[1] * TIME_STEP
                new_x = np.clip(new_x, 0.5, self.grid_width - 0.5)
                new_y = np.clip(new_y, 0.5, self.grid_height - 0.5)
                target = (ped.floor, int(round(new_x)), int(round(new_y)))

                present = (ped.floor, int(round(ped.x)), int(round(ped.y)))
                if target not in occupied and not self.environment.is_blocked(ped.x, ped.y, new_x, new_y):
                    ped.x, ped.y = new_x, new_y
                    occupied.discard(present)  # Remove old position
                    occupied.add(target)

                # Ensure velocity is non-zero
                if np.linalg.norm(ped.velocity) < 1e-3:
                    ped.velocity = np.random.uniform(-1, 1, 2)
                    ped.velocity /= np.linalg.norm(ped.velocity)
                    ped.velocity *= ped.desired_speed

                ped.z = ped.floor * (self.environment.floor_height + self.environment.floor_thickness)

                # Check for stair proximity
                if ped.floor > 0 and not ped.on_stairs:
                    stair = self.environment.get_stair_from(ped.floor, ped.x, ped.y)
                    if stair and abs(ped.x - stair.start[0]) < 2 and abs(ped.y - stair.start[1]) < 2:
                        ped.on_stairs = True
                        ped.current_stair = stair
                        ped.stair_step_idx = 0

                # Exit check (queueing)
                handle_exit_queueing(ped, exit_queues, evac_time, self.environment)

                # Handle door queueing if at a door (before movement)
                if door_queues is not None:
                    handle_door_queueing(ped, door_queues, self.environment)

            except Exception as e:
                print(f"[DEBUG][MFO_SFM] Error processing ped {getattr(ped, 'id', None)}: {e}")

        evacuated_peds += process_exit_queues(exit_queues, evac_time)

        # After all movement, process door queues
        if door_queues is not None:
            process_door_queues(door_queues, self.environment)

        print(f"MFO_SFM : Evacuated {evacuated_peds} pedestrians out of {len(self.pedestrians)}")
        return evacuated_peds
