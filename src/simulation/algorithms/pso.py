import numpy as np
import random
from param_config import TIME_STEP, GRID_LENGTH, GRID_WIDTH, DOOR_WIDTH,DESIRED_SPEED, QUEUE_RECTANGLE_LENGTH, QUEUE_RECTANGLE_WIDTH

class PSOMover:
    def __init__(self, pedestrians, environment, max_iter=200, inertia_weight=0.5, cognitive_coeff=1.5, social_coeff=1.5):
        self.pedestrians = pedestrians
        # Set all as staff (pure PSO agents)
        for ped in self.pedestrians:
            ped.is_staff = True
        self.environment = environment
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_LENGTH
        self.max_iter = max_iter
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff
        self.exit_points = environment.get_exit_locations()
        self.g_best = None  # Initialize global best position

    def move(self, time_step):
        evacuated_peds = 0
        occupied = set()
        occupied_stairs = set()
        for ped in self.pedestrians:
            if getattr(ped, 'evacuated', False):
                continue
            # Mark occupied positions
            if ped.on_stairs:
                occupied_stairs.add((id(ped.current_stair), ped.stair_step_idx))
            else:
                occupied.add((ped.floor, int(round(ped.x)), int(round(ped.y))))

        # Update global best position using environment's evacuation_heuristic
        for ped in self.pedestrians:
            goal = self.environment.get_nearest_goal(ped)
            # For exit heuristic, get exit_room and exit_floor
            if ped.floor == 0:
                nearest_exit = self.environment.get_nearest_exit(ped.x, ped.y)
                exit_room = self.environment.get_room(nearest_exit['x'], nearest_exit['y'])
                exit_floor = 0
            else:
                stair = self.environment.get_stair_from(ped.floor, ped.x, ped.y)
                exit_room = self.environment.get_room(stair.start[0], stair.start[1]) if stair else None
                exit_floor = ped.floor
            ped_best = (ped.x, ped.y)
            ped_best_score = self.environment.evacuation_heuristic(ped, np.array([ped.x, ped.y]), exit_room=exit_room, exit_floor=exit_floor)
            if self.g_best is None:
                self.g_best = ped_best
                self.g_best_score = ped_best_score
            else:
                if ped_best_score < self.g_best_score:
                    self.g_best = ped_best
                    self.g_best_score = ped_best_score

        for ped in self.pedestrians:
            if getattr(ped, 'evacuated', False):
                evacuated_peds += 1
                continue

            # Handle stair movement using shared utility
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

            # Use get_nearest_goal for PSO agents (staff-aware)
            goal = self.environment.get_nearest_goal(ped, staff_aware=True)
            if goal is not None:
                gx, gy = goal[0], goal[1]
            else:
                gx, gy = ped.x, ped.y

            r1, r2 = random.random(), random.random()
            ped.vx = self.w * ped.vx + self.c1 * r1 * (ped.px_best - ped.x) + self.c2 * r2 * (gx - ped.x)
            ped.vy = self.w * ped.vy + self.c1 * r1 * (ped.py_best - ped.y) + self.c2 * r2 * (gy - ped.y)
            
            # Incorporate global best position in velocity calculation
            if self.g_best:
                ped.vx += self.c2 * random.random() * (self.g_best[0] - ped.x)
                ped.vy += self.c2 * random.random() * (self.g_best[1] - ped.y)

            speed = np.linalg.norm(ped.velocity)
            if speed > ped.max_speed:
                scale = ped.desired_speed / speed
                ped.vx *= scale
                ped.vy *= scale

            new_x = min(max(ped.x + ped.vx, 0), self.grid_width - 1)
            new_y = min(max(ped.y + ped.vy, 0), self.grid_height - 1)
            target = (ped.floor, int(round(new_x)), int(round(new_y)))
            # BLOCKING CONDITION: Only move if not blocked by wall
            if target not in occupied and not self.environment.is_blocked(ped.x,ped.y,new_x, new_y):
                ped.x, ped.y = new_x, new_y
                occupied.add(target)
            # else: stay in place

            ped.z = ped.floor * (self.environment.floor_height+ self.environment.floor_thickness / 2)

            # Update personal best
            if self._distance(ped.x, ped.y, gx, gy) < self._distance(ped.px_best, ped.py_best, gx, gy):
                ped.px_best, ped.py_best = ped.x, ped.y

            # Check for stairs
            stair = self.environment.get_stair_from(ped.floor, ped.x, ped.y)
            if stair and abs(ped.x - stair.start[0]) < 2.5 and abs(ped.y - stair.start[1]) < 2.5:
                ped.on_stairs = True
                ped.current_stair = stair
                ped.stair_step_idx = 0

            # Check for exit on ground floor using rectangle-based zone
            if ped.floor == 0:
                exit = self.environment.get_nearest_exit(ped.x, ped.y)
                gx_exit, gy_exit = exit['x'], exit['y']
                if (gx_exit - QUEUE_RECTANGLE_LENGTH / 2 <= ped.x <= gx_exit + QUEUE_RECTANGLE_LENGTH / 2 and
                    gy_exit <= ped.y <= gy_exit + QUEUE_RECTANGLE_WIDTH and
                    not getattr(ped, 'evacuated', False)):
                    ped.mark_evacuated(time_step)
                    evacuated_peds += 1

        print(f"PSO : Evacuated {evacuated_peds} pedestrians out of {len(self.pedestrians)}")
        return evacuated_peds

    def _distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
