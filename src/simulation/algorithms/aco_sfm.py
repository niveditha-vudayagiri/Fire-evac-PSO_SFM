import numpy as np
import random
from param_config import (GRID_WIDTH, GRID_LENGTH, TIME_STEP, PED_RADIUS, tau, mass, STAFF_INFLUENCE_RADIUS, DOOR_WIDTH)
from .sfm_common import calculate_social_force, calculate_wall_force, process_exit_queues, handle_exit_queueing

class ACOSFMMover:
    """
    Ant Colony Optimization inspired SFM mover.
    Pedestrians use visual cues and word-of-mouth (pheromone analogy) to evacuate optimally.
    """
    def __init__(self, pedestrians, environment, pheromone_decay=0.1, pheromone_strength=1.0):
        self.pedestrians = pedestrians
        self.environment = environment
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_LENGTH
        self.tau = tau
        self.mass = mass
        self.pheromone_decay = pheromone_decay
        self.pheromone_strength = pheromone_strength
        self.pheromone_map = np.zeros((GRID_WIDTH, GRID_LENGTH))  # Pheromone intensity grid

    def _update_pheromones(self):
        """Decay pheromones and update based on pedestrian movement."""
        self.pheromone_map *= (1 - self.pheromone_decay)  # Apply decay
        for ped in self.pedestrians:
            if not getattr(ped, 'evacuated', False):
                x, y = int(round(ped.x)), int(round(ped.y))
                if 0 <= x < self.pheromone_map.shape[0] and 0 <= y < self.pheromone_map.shape[1]:
                    self.pheromone_map[x, y] += self.pheromone_strength

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

        # Update pheromones
        self._update_pheromones()

        for ped in self.pedestrians:
            try:
                if getattr(ped, 'evacuated', False):
                    evacuated_peds += 1
                    continue

                # Handle stair movement
                if getattr(ped, 'on_stairs', False):
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

                pos = np.array([ped.x, ped.y])

                # Determine goal (exit or stairs)
                goal = self.environment.get_nearest_goal(ped)
                
                # Calculate pheromone influence
                pheromone_influence = np.zeros(2)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = int(round(ped.x + dx)), int(round(ped.y + dy))
                        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_LENGTH:
                            pheromone_influence += self.pheromone_map[nx, ny] * np.array([dx, dy])

                # Normalize pheromone influence
                pheromone_norm = np.linalg.norm(pheromone_influence)
                if pheromone_norm > 0:
                    pheromone_influence /= pheromone_norm

                # Combine pheromone influence with goal direction
                if goal is not None:
                    goal_vec = goal - pos
                    dist_goal = np.linalg.norm(goal_vec)
                    if dist_goal > 0:
                        goal_direction = goal_vec / dist_goal
                    else:
                        goal_direction = np.zeros(2)
                else:
                    goal_direction = np.zeros(2)

                desired_direction = 0.7 * goal_direction + 0.3 * pheromone_influence
                desired_direction_norm = np.linalg.norm(desired_direction)
                if desired_direction_norm > 0:
                    desired_direction /= desired_direction_norm

                v_desired = ped.desired_speed * desired_direction
                ped.desired_direction = desired_direction

                # Ensure velocity is a numpy array
                if not hasattr(ped, 'velocity') or not isinstance(ped.velocity, np.ndarray):
                    ped.velocity = np.zeros(2)
                f_drive = (v_desired - ped.velocity) / self.tau
                from .sfm_common import amplify_drive_if_in_queue
                f_drive = amplify_drive_if_in_queue(f_drive, ped)
                # Calculate repulsion and wall forces
                f_repulsion = calculate_social_force(ped, pos, self.pedestrians)
                f_wall = calculate_wall_force(ped, pos, self.environment, self.grid_width, self.grid_height)

                # Update velocity and position
                force = f_drive + (f_repulsion + f_wall) / self.mass
                ped.velocity += force * TIME_STEP

                # Dampen speed if in queue
                if getattr(ped, '_in_exit_queue', False) or getattr(ped, '_in_door_queue', False):
                    ped.velocity *= 0.7  # Damping factor for queueing

                # Clip velocity to reasonable speed
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
                    from .sfm_common import handle_door_queueing
                    handle_door_queueing(ped, door_queues, self.environment)

            except Exception as e:
                print(f"[DEBUG][ACO_SFM] Error processing ped {getattr(ped, 'id', None)}: {e}")

        evacuated_peds += process_exit_queues(exit_queues, evac_time)
        print(f"ACO_SFM : Evacuated {evacuated_peds} pedestrians out of {len(self.pedestrians)}")

        # After all movement, process door queues
        if door_queues is not None:
            from .sfm_common import process_door_queues
            process_door_queues(door_queues, self.environment)

        return evacuated_peds
