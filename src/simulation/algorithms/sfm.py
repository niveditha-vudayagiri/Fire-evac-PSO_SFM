import numpy as np
from param_config import GRID_LENGTH, GRID_WIDTH, TIME_STEP, PED_RADIUS, A, B, k, kappa, tau, mass,DOOR_WIDTH, VISIBILITY
from .sfm_common import calculate_social_force, calculate_wall_force, process_exit_queues, handle_exit_queueing
import queue

class SFMMover:
    def __init__(self, pedestrians, environment):
        self.pedestrians = pedestrians
        self.environment = environment
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_LENGTH


    def move(self, evac_time, door_queues=None):
        evacuated_peds = 0

        occupied = set()
        occupied_stairs = set()

        for ped in self.pedestrians:
            if getattr(ped, 'evacuated', False):
                continue

            if ped.on_stairs:
                occupied_stairs.add((id(ped.current_stair), ped.stair_step_idx))
            else:
                occupied.add((ped.floor, int(round(ped.x)), int(round(ped.y))))

        # --- Exit queue logic ---
        # For each exit, maintain a queue of (pedestrian, arrival_time)
        exit_queues = {}
        for exit in self.environment.exits:
            exit_queues[(exit['x'], exit['y'])] = []

        # First, sort pedestrians by their proximity to exits (for fairness)
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
                    ped.x = np.clip(ped.x, 0, self.grid_width - 1)
                    ped.y = np.clip(ped.y, 0, self.grid_height - 1)
                    continue

                # Determine goal (exit or stairs)
                goal = self.environment.get_nearest_goal(ped)

                pos = np.array([ped.x, ped.y])
                if goal is not None:
                    # Driving force towards goal (exit or stairs)
                    goal_vec = goal - pos
                    dist_goal = np.linalg.norm(goal_vec)
                    if dist_goal > 0:
                        e0 = goal_vec / dist_goal  # normalize: unit vector
                    else:
                        e0 = np.zeros(2)
                else:
                    # Neighbor alignment (herding behavior)
                    neighbor_velocities = []
                    if VISIBILITY:
                        for other in self.pedestrians:
                            if other is ped or getattr(other, 'evacuated', False) or ped.floor != other.floor:
                                continue
                            dist = np.linalg.norm(pos - np.array([other.x, other.y]))
                            if dist < 3.0:
                                neighbor_velocities.append(getattr(other, 'velocity', np.zeros(2)))
                    
                    if neighbor_velocities and VISIBILITY:
                        avg_velocity = np.mean(neighbor_velocities, axis=0)
                        norm = np.linalg.norm(avg_velocity)
                        e0 = avg_velocity / norm if norm > 0 else np.zeros(2)
                    else:
                        random_direction = np.random.uniform(-1, 1, 2)
                        e0 = random_direction / np.linalg.norm(random_direction)

                v_desired = ped.desired_speed * e0
                ped.desired_direction = e0

                # Ensure velocity is a numpy array
                if not hasattr(ped, 'velocity') or not isinstance(ped.velocity, np.ndarray):
                    ped.velocity = np.zeros(2)
                f_drive = (v_desired - ped.velocity) / tau
                from .sfm_common import amplify_drive_if_in_queue
                f_drive = amplify_drive_if_in_queue(f_drive, ped)
                # Calculate repulsion and wall forces
                f_repulsion = calculate_social_force(ped, pos, self.pedestrians)
                f_wall = calculate_wall_force(ped, pos, self.environment, self.grid_width, self.grid_height)

                # Update velocity and position
                force = f_drive + (f_repulsion + f_wall) / mass
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
                print(f"[DEBUG][SFM] Error processing ped {getattr(ped, 'id', None)}: {e}")

        # Process exit queues: allow only max_side_by_side to evacuate per time step per exit (FIFO, only at the door)
        evacuated_peds += process_exit_queues(exit_queues, evac_time)

        # After all movement, process door queues
        if door_queues is not None:
            from .sfm_common import process_door_queues
            process_door_queues(door_queues, self.environment)

        print(f"SFM : Evacuated {evacuated_peds} pedestrians out of {len(self.pedestrians)}")
        return evacuated_peds
