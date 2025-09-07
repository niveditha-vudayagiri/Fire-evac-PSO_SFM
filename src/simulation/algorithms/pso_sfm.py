import numpy as np
import random
from param_config import (GRID_WIDTH, GRID_LENGTH, TIME_STEP, PED_RADIUS, 
                        A, B, k, kappa, tau, mass,
                        COMPLIANCE_RATE_MIN, COMPLIANCE_RATE_MAX, STAFF_INFLUENCE_RADIUS, N_PEDESTRIANS,DOOR_WIDTH,VISIBILITY)
from .sfm_common import calculate_social_force, calculate_wall_force, process_exit_queues, handle_exit_queueing, handle_door_queueing, process_door_queues

class PSOSFMMover:
    def __init__(self, pedestrians, environment, staff_ratio=0.05, staff_policy=None):
        self.pedestrians = pedestrians
        self.environment = environment
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_LENGTH
        self.tau = tau  # relaxation time in seconds
        self.staff_ratio = staff_ratio
        self.staff_policy = staff_policy  # Store the staff policy for use in logic

        # PSO parameters
        self.w_pso = 0.5      # PSO inertia weight
        self.c1 = 1.5         # cognitive coefficient
        self.c2 = 1.5         # social coefficient
        self.A = A            # interaction strength
        self.B = B            # interaction range
        self.K = k            # body force constant
        self.KAPPA = kappa    # sliding friction
        self.MASS = mass      # average mass of a pedestrian in kg

        # Assign staff agents as leaders and their strategic positions
        # Identify staff agents based on is_staff attribute
        staff_indices = [i for i, ped in enumerate(self.pedestrians) if getattr(ped, 'is_staff', False)]
        n_staff = len(staff_indices)
        assisting_indices = set()

        #Set default staff floor priority
        for i in staff_indices:
            self.pedestrians[i].staff_floor_priority = list(range(self.environment.floor_count))

        # Assign staff_priority (lower value = higher priority)
        for i in staff_indices:
            self.pedestrians[i].staff_priority = i  # You can customize this assignment for more complex logic

        if self.staff_policy == 'zone_sweep':
            # Assign staff to rooms based on environment room coordinates
            for i in staff_indices:
                self.pedestrians[i].assigned_rooms = self.environment.get_room(self.pedestrians[i].x, self.pedestrians[i].y, self.pedestrians[i].floor)
                self.pedestrians[i].staff_floor_priority = list(range(self.environment.floor_count))
        elif self.staff_policy == 'top_evac':
            for i in staff_indices:
                self.pedestrians[i].staff_floor_priority = list(reversed(range(self.environment.floor_count)))
        elif self.staff_policy == 'avoid_top':
            for i in staff_indices:
                self.pedestrians[i].staff_floor_priority = list(range(self.environment.floor_count - 1))
        
        if self.staff_policy == 'half_assist_half_leave':
            # Randomly split staff into assisting and evacuating
            n_assist = n_staff // 2
            assisting_indices = set(np.random.choice(list(staff_indices), n_assist, replace=False))
        
        for i, ped in enumerate(self.pedestrians):
            if ped.is_staff:
                ped.in_position = True  # Staff influence immediately
                ped.compliance_rate = np.random.uniform(COMPLIANCE_RATE_MIN, COMPLIANCE_RATE_MAX)
                ped.assigned_zone = None
                # Assign staff role for half_assist_half_leave
                if self.staff_policy == 'half_assist_half_leave':
                    ped.staff_assist = (i in assisting_indices)
                elif self.staff_policy == 'all_evacuate':
                    ped.staff_assist = False
                    ped.evacuating = True
                    ped.goal = self.environment.get_nearest_goal(ped, staff_aware=True, is_pso=True)
                    # Set higher desired speed for evacuating staff
                    ped.desired_speed *= 1.2
                    # Initialize velocity towards exit
                    pos = np.array([ped.x, ped.y])
                    direction = ped.goal - pos
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        ped.velocity = (direction / norm) * ped.desired_speed
                    else:
                        ped.velocity = np.zeros(2)
                else:
                    ped.staff_assist = True  # Default: staff assist unless policy says otherwise
                # --- PSO agent state ---
                # Always ensure pso_velocity is a valid vector for staff
                if not hasattr(ped, 'pso_velocity') or not isinstance(ped.pso_velocity, np.ndarray) or np.linalg.norm(ped.pso_velocity) < 1e-3:
                    initial_goal = self._assign_staff_goal(ped)
                    pos = np.array([ped.x, ped.y])
                    if initial_goal is not None and isinstance(initial_goal, np.ndarray) and initial_goal.shape == (2,):
                        direction = initial_goal - pos
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            ped.pso_velocity = (direction / norm) * getattr(ped, 'desired_speed', 1.0)
                        else:
                            ped.pso_velocity = np.random.uniform(-1, 1, 2)
                            ped.pso_velocity /= np.linalg.norm(ped.pso_velocity)
                            ped.pso_velocity *= getattr(ped, 'desired_speed', 1.0)
                    else:
                        ped.pso_velocity = np.random.uniform(-1, 1, 2)
                        ped.pso_velocity /= np.linalg.norm(ped.pso_velocity)
                        ped.pso_velocity *= getattr(ped, 'desired_speed', 1.0)
                if not hasattr(ped, 'px_best'):
                    ped.px_best = ped.x
                if not hasattr(ped, 'py_best'):
                    ped.py_best = ped.y

        # Initialize px_best/py_best for all pedestrians
        for ped in self.pedestrians:
            if not hasattr(ped, 'px_best'):
                ped.px_best = ped.x
            if not hasattr(ped, 'py_best'):
                ped.py_best = ped.y
            # Assign compliance_rate to civilians (not staff)
            if not getattr(ped, 'is_staff', False):
                if not hasattr(ped, 'compliance_rate'):
                    ped.compliance_rate = np.random.uniform(COMPLIANCE_RATE_MIN, COMPLIANCE_RATE_MAX)

    def is_near_staff(self, civ, staff_positions):
        """Check if civilian is in the same room as any staff member."""
        civ_room = self.environment.get_room(civ.x, civ.y, civ.floor)
        if not civ_room:
            return False
        
        for staff_pos in staff_positions:
            staff_room = self.environment.get_room(staff_pos[0], staff_pos[1], civ.floor)
            if staff_room and staff_room == civ_room and civ.compliance_rate > 0.5:
                return True
        return False

    def move(self, evac_time, door_queues=None):
        evacuated_peds = 0

        # Initialize exit queues (same as SFM)
        max_side_by_side = int(DOOR_WIDTH // (2 * PED_RADIUS))
        exit_queues = {}
        for exit in self.environment.exits:
            exit_queues[(exit['x'], exit['y'])] = []

        try:
            # Track occupied positions
            occupied = set()
            occupied_stairs = set()
            for ped in self.pedestrians:
                if getattr(ped, 'evacuated', False):
                    evacuated_peds += 1
                    continue
                if ped.on_stairs:
                    occupied_stairs.add((id(ped.current_stair), ped.stair_step_idx))
                else:
                    occupied.add((ped.floor, int(round(ped.x)), int(round(ped.y))))

            """#TEMP - Staff Leave Midway
            if evac_time == 100:
                #Set all staff to evacuate
                for ped in self.pedestrians:
                    if getattr(ped, 'is_staff', False):
                        ped.evacuating = True
                        ped.goal = self.environment.get_nearest_goal(ped, staff_aware=True, is_pso=True)"""

            # Precompute staff and civilians
            staff = [p for p in self.pedestrians if getattr(p, 'is_staff', False) and not getattr(p, 'evacuated', False)]
            civilians = [p for p in self.pedestrians if not getattr(p, 'is_staff', False) and not getattr(p, 'evacuated', False)]
            staff_by_floor = {}
            for s in staff:
                staff_by_floor.setdefault(s.floor, []).append(s)

            
            # Precompute policy-based priority group for civilians (who can be influenced)
            priority_group = None
            staff_positions = [np.array([s.x, s.y]) for s in staff]
            if self.staff_policy == 'zone_sweep':
                # For zone_sweep, each staff influences civilians in their assigned rooms and same room
                priority_group = set()
                for s in staff:
                    # Use assigned_rooms for each staff
                    staff_rooms = getattr(s, 'assigned_rooms', None)
                    if not staff_rooms:
                        # Fallback: assign current room if not set
                        staff_rooms = [self.environment.get_room(s.x, s.y, s.floor)]
                        s.assigned_rooms = staff_rooms  # Ensure assigned_rooms is set

                    zone_civilians = [
                        c for c in civilians
                        if (not getattr(c, 'evacuated', False)
                            and c.floor == s.floor
                            and self.environment.get_room(c.x, c.y, c.floor) == staff_rooms
                            and c.compliance_rate > 0.5)  # Only include compliant civilians
                    ]
                    priority_group.update(zone_civilians)

            elif self.staff_policy == 'assist_mobile_first':
                mobile = [c for c in civilians if c.age_group!="elderly" and self.is_near_staff(c, staff_positions)]
                if len(mobile) > 0:
                    priority_group = set(mobile)
                else:
                    others = [c for c in civilians if self.is_near_staff(c, staff_positions)]
                    if len(others)>0:
                        priority_group = set(others)
            elif self.staff_policy == 'assist_elderly_first':
                elderly = [c for c in civilians if c.age_group=="elderly" and self.is_near_staff(c, staff_positions)]
                if len(elderly) > 0:
                    priority_group = set(elderly)
                else:
                    mobile = [c for c in civilians if self.is_near_staff(c, staff_positions)]
                    if len(mobile)>0:
                        priority_group = set(mobile)
            elif self.staff_policy != 'all_evacuate':
                priority_group = set([c for c in civilians if self.is_near_staff(c, staff_positions)])

            # Unify staff goal assignment and direction
            staff_guidance_goals = {}
            staff_influence = {}
            for ped in staff:
                pos = np.array([ped.x, ped.y])
                # Assign goal based on policy (where staff moves)
                guidance_goal = self._assign_staff_goal(ped)
                if guidance_goal is None or not (isinstance(guidance_goal, np.ndarray) and guidance_goal.shape == (2,)):
                    print(f"[DEBUG][PSO_SFM] Invalid goal for staff {getattr(ped, 'id', None)}: {guidance_goal}. Using current position.")
                    guidance_goal = pos.copy()

                # Assign guidance_goal: where staff tells civilians to go
                # By default, guidance_goal is the nearest exit for the staff
                ped.guidance_goal = guidance_goal
                staff_guidance_goals[ped] = guidance_goal

                # Store staff position, room, and floor information
                staff_room = self.environment.get_room(ped.x, ped.y, ped.floor)
                staff_influence[ped] = (np.array([ped.x, ped.y]), staff_room, ped.floor)

            # Coordinate staff movement and zone assignments (only if assisting)
            self._coordinate_staff()

            # 1. Move staff (update positions if allowed by policy)
            for ped in staff:
                # --- PSO update for staff ---
                goal = ped.goal
                pos = np.array([ped.x, ped.y])
                r1, r2 = random.random(), random.random()
                # More aggressive PSO parameters for staff in all_evacuate mode
                w = 0.8 if self.staff_policy == 'all_evacuate' else self.w_pso
                c1 = 2.0 if self.staff_policy == 'all_evacuate' else self.c1
                c2 = 2.0 if self.staff_policy == 'all_evacuate' else self.c2
                # Personal best update
                if np.linalg.norm(pos - goal) < np.linalg.norm(np.array([ped.px_best, ped.py_best]) - goal):
                    ped.px_best, ped.py_best = ped.x, ped.y
                # PSO velocity update with improved goal-seeking
                cognitive = c1 * r1 * (np.array([ped.px_best, ped.py_best]) - pos)
                social = c2 * r2 * (goal - pos)
                # Normalize components to maintain influence
                if np.linalg.norm(cognitive) > 0:
                    cognitive = cognitive / np.linalg.norm(cognitive) * ped.desired_speed
                if np.linalg.norm(social) > 0:
                    social = social / np.linalg.norm(social) * ped.desired_speed
                # Update PSO velocity robustly
                if not hasattr(ped, 'pso_velocity') or not isinstance(ped.pso_velocity, np.ndarray) or np.linalg.norm(ped.pso_velocity) < 1e-3:
                    ped.pso_velocity = np.random.uniform(-1, 1, 2)
                    ped.pso_velocity /= np.linalg.norm(ped.pso_velocity)
                    ped.pso_velocity *= getattr(ped, 'desired_speed', 1.0)
                ped.pso_velocity = w * ped.pso_velocity + cognitive + social
                # Clip velocity to max speed
                speed = np.linalg.norm(ped.pso_velocity)
                if speed > getattr(ped, 'desired_speed', 1.0) * 1.5:
                    ped.pso_velocity = ped.pso_velocity * (getattr(ped, 'desired_speed', 1.0) / speed)
                # If staff velocity is too low, kickstart with pso_velocity
                if np.linalg.norm(ped.velocity) < 1e-3:
                    ped.velocity = ped.pso_velocity.copy()

                #For immobile staff, comment out the evacuation logic
                evacuated = self._move_staff(
                    ped, occupied, occupied_stairs, exit_queues,
                    door_queues, evac_time
                )
                
            # 3. Move civilians, referencing staff goals if within influence of non-evacuating staff
            for ped in civilians:
                evacuated = self._move_civilian(
                    ped, civilians, staff_influence, staff_guidance_goals, priority_group,
                    occupied, occupied_stairs, exit_queues,
                    door_queues, evac_time
                )

            # Use shared exit queue processing
            if door_queues is not None:
                process_door_queues(door_queues, self.environment)
            if exit_queues is not None:
                evacuated_peds += process_exit_queues(exit_queues, evac_time)

        except Exception as e:
            print(f"[DEBUG][PSO_SFM] Error in move(): {e}")

        print(f"PSO_SFM : Evacuated {evacuated_peds} pedestrians out of {len(self.pedestrians)}")
        # After all movement, process door queues
        return evacuated_peds

    def _coordinate_staff(self):
        """Coordinate staff positions and properties based on policy and building state"""
        # For all_evacuate policy, ensure all staff are evacuating
        if self.staff_policy == 'all_evacuate':
            for ped in self.pedestrians:
                if getattr(ped, 'is_staff', False) and not getattr(ped, 'evacuated', False):
                    ped.evacuating = True
                    ped.staff_assist = False
                    # Update goal to nearest exit
                    ped.goal = self.environment.get_nearest_goal(ped, staff_aware=True, is_pso=True)
            return

        active_staff = [ped for ped in self.pedestrians 
                        if ped.is_staff and 
                           not ped.evacuated]
        
        # Get count of civilians per floor for efficient checking
        civilians_by_floor = {}
        for ped in self.pedestrians:
            if not ped.is_staff and not ped.evacuated:
                civilians_by_floor[ped.floor] = civilians_by_floor.get(ped.floor, 0) + 1

        for staff in active_staff:

            #Default staff properties
            staff.goal = self.environment.get_nearest_goal(staff, staff_aware=True, is_pso=True)

            current_floor = staff.floor
            current_room = self.environment.get_room(staff.x, staff.y, current_floor)
            
            if getattr(staff, 'evacuating', False):
                continue

            # Check if current floor is empty (no civilians)
            if civilians_by_floor.get(current_floor, 0) == 0:
                staff.evacuating = True
                staff.goal = self.environment.get_nearest_goal(staff, staff_aware=True, is_pso=True)
                staff.following = False
                continue

            # Policy-specific coordination
            if self.staff_policy == 'zone_sweep':
                x_min, y_min, x_max, y_max = staff.assigned_rooms
                # Find civilians in this zone and floor
                zone_civilians = [p for p in self.pedestrians 
                                if not p.evacuated and not p.is_staff 
                                and x_min <= p.x <= x_max 
                                and y_min <= p.y <= y_max
                                and p.floor == current_floor]
                if zone_civilians:
                    # Assist nearest civilian in zone
                    nearest = min(zone_civilians, key=lambda p: (p.x - staff.x)**2 + (p.y - staff.y)**2)
                    staff.goal = np.array([nearest.x, nearest.y])
                    staff.following = True
                    staff.staff_assist = True
                    staff.evacuating = False
                else:
                    # Zone is empty, evacuate and prioritize self
                    staff.evacuating = True
                    staff.staff_assist = False
                    staff.goal = self.environment.get_nearest_goal(staff, staff_aware=True, is_pso=True)
                    staff.following = False

            elif self.staff_policy in ['top_evac', 'avoid_top']:
                # Ensure floor priorities are set
                if not hasattr(staff, 'staff_floor_priority'):
                    if self.staff_policy == 'top_evac':
                        staff.staff_floor_priority = list(reversed(range(self.environment.floor_count)))
                    else:  # avoid_top
                        staff.staff_floor_priority = list(range(self.environment.floor_count - 1))
                        if(staff.floor == self.environment.floor_count - 1):
                            staff.evacuating = True
                
                # Find highest priority floor that still has civilians
                target_floor = None
                for f in staff.staff_floor_priority:
                    if civilians_by_floor.get(f, 0) > 0:
                        target_floor = f
                        break
                
                if target_floor is None or target_floor == current_floor:
                    # Current floor is highest priority with civilians
                    civilians_here = [p for p in self.pedestrians 
                                    if not p.evacuated and not p.is_staff 
                                    and p.floor == current_floor]
                    if civilians_here:
                        nearest = min(civilians_here, key=lambda p: (p.x - staff.x)**2 + (p.y - staff.y)**2)
                        staff.goal = np.array([nearest.x, nearest.y])
                        staff.following = True
                    else:
                        # No civilians left on any priority floor, evacuate
                        staff.evacuating = True
                        staff.goal = self.environment.get_nearest_goal(staff, staff_aware=True, is_pso=True)
                        staff.following = False
                else:
                    # Move to higher priority floor
                    stair = self.environment.get_stair_to(current_floor, target_floor)
                    if stair:
                        staff.goal = np.array(stair.start)
                        staff.target_floor = target_floor

            elif self.staff_policy == 'all_assist':
                # Find highest density area on current floor
                density_map = {}
                for ped in self.pedestrians:
                    if not ped.is_staff and not ped.evacuated and ped.floor == current_floor:
                        room = self.environment.get_room(ped.x, ped.y, ped.floor)
                        if room:
                            density_map[room] = density_map.get(room, 0) + 1
                
                if density_map:
                    # Target highest density room
                    target_room = max(density_map, key=density_map.get)
                    room_bounds = self.environment.get_room_bounds(target_room)
                    if room_bounds:
                        center_x = (room_bounds[0] + room_bounds[2]) / 2
                        center_y = (room_bounds[1] + room_bounds[3]) / 2
                        staff.goal = np.array([center_x, center_y])
                        staff.following = True
                        staff.staff_assist = True
                else:
                    # No civilians in any room on this floor, evacuate
                    staff.evacuating = True
                    staff.goal = self.environment.get_nearest_exit(staff)
                    staff.following = False

            elif self.staff_policy == 'half_assist_half_leave':
                # Ensure staff_assist property is set
                if not hasattr(staff, 'staff_assist'):
                    n_staff = len(active_staff)
                    n_assist = n_staff // 2
                    staff.staff_assist = active_staff.index(staff) < n_assist

                if staff.staff_assist:
                    # Assisting staff behavior (similar to all_assist)
                    civilians_here = [p for p in self.pedestrians 
                                    if not p.evacuated and not p.is_staff 
                                    and p.floor == current_floor]
                    if civilians_here:
                        nearest = min(civilians_here, key=lambda p: (p.x - staff.x)**2 + (p.y - staff.y)**2)
                        staff.goal = np.array([nearest.x, nearest.y])
                        staff.following = True
                    else:
                        staff.evacuating = True
                        staff.goal = self.environment.get_nearest_goal(staff, staff_aware=True, is_pso=True)
                        staff.following = False
                else:
                    # Non-assisting staff should evacuate
                    staff.evacuating = True
                    staff.goal = self.environment.get_nearest_goal(staff, staff_aware=True, is_pso=True)
                    staff.following = False
            

    def _blend_with_staff_direction(self, civilian_dir, staff_dir, congestion_ratio, following=False):
        """
        Blend the civilian's desired direction with the staff's direction.
        If following staff, give higher priority to staff_dir regardless of congestion.
        Args:
            civilian_dir (np.ndarray): Civilian's original desired direction.
            staff_dir (np.ndarray): Staff's direction.
            congestion_ratio (float): Value between 0 and 1 indicating congestion.
            following (bool): If True, always prioritize staff_dir (alpha=0.8).
        Returns:
            np.ndarray: Blended direction vector.
        """
        if following:
            alpha = 0.8  # High priority to staff_dir if following
        else:
            alpha = min(max(congestion_ratio, 0.0), 1.0)
        blended = (1 - alpha) * civilian_dir + alpha * staff_dir
        norm = np.linalg.norm(blended)
        if norm > 0:
            return blended / norm
        else:
            return civilian_dir
        
    def _move_staff(self, ped, occupied, occupied_stairs, exit_queues, door_queues, evac_time):
        try:
            if getattr(ped, 'evacuated', False):
                return True
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
                    return False
                except Exception as e:
                    print(f"[DEBUG][PSO_SFM] Error moving staff on stairs: {e}")
                    return False
            pos = np.array([ped.x, ped.y])
            if ped.pso_velocity is not None and isinstance(ped.pso_velocity, np.ndarray) and np.linalg.norm(ped.pso_velocity) > 1e-3:
                ped.velocity = ped.pso_velocity

            # Get current goal or update if needed
            if not hasattr(ped, 'goal') or ped.goal is None:
                ped.goal = self.environment.get_nearest_goal(ped, staff_aware=True)
            
            # Direct movement towards exit with minimal interference
            goal_direction = ped.goal - pos
            
            distance = np.linalg.norm(goal_direction)
            desired_speed = (goal_direction / distance) * ped.desired_speed * 1.2
            # Strong goal-directed force with minimal social forces
            f_goal = (desired_speed - ped.velocity) / self.tau
            f_repulsion = calculate_social_force(ped, pos, self.pedestrians)
            f_wall = calculate_wall_force(ped, pos, self.environment, self.grid_width, self.grid_height)
            
            if distance > 0:
                total_force = f_goal + (f_repulsion + f_wall) / self.MASS
            else:
                total_force = (f_repulsion + f_wall) / self.MASS

            # Update velocity
            ped.velocity = ped.velocity + total_force * TIME_STEP

            # Dampen speed if in queue
            if getattr(ped, '_in_exit_queue', False) or getattr(ped, '_in_door_queue', False):
                ped.velocity *= 0.7  # Damping factor for queueing

            speed = np.linalg.norm(ped.velocity)
            if speed > ped.max_speed:
                ped.velocity *= (ped.desired_speed / speed)

            # Update position
            new_x = ped.x + ped.velocity[0] * TIME_STEP
            new_y = ped.y + ped.velocity[1] * TIME_STEP
            new_x = np.clip(new_x, 0.5, self.grid_width - 0.5)
            new_y = np.clip(new_y, 0.5, self.grid_height - 0.5)

            # Track movement history for stuck detection
            if not hasattr(ped, '_pos_history'):
                ped._pos_history = []
            ped._pos_history.append((ped.x, ped.y))
            if len(ped._pos_history) > 10:
                ped._pos_history.pop(0)

            # Check if stuck (hasn't moved significantly in last 10 steps)
            if len(ped._pos_history) == 10:
                dists = [np.linalg.norm(np.array([ped.x, ped.y]) - np.array(pos)) for pos in ped._pos_history]
                if max(dists) < 0.5:
                    # Nudge with a random direction
                    nudge = np.random.uniform(-1, 1, 2)
                    nudge /= np.linalg.norm(nudge)
                    ped.velocity += 0.5 * ped.desired_speed * nudge

            # Check occupancy and update position
            target = (ped.floor, int(round(new_x)), int(round(new_y)))
            present = (ped.floor, int(round(ped.x)), int(round(ped.y)))
            if target not in occupied and not self.environment.is_blocked(ped.x, ped.y, new_x, new_y):
                ped.x, ped.y = new_x, new_y
                occupied.discard(present)
                occupied.add(target)
            
            # Add small random movement if stuck
            if np.linalg.norm(ped.velocity) < 1e-3:
                random_dir = np.random.uniform(-1, 1, 2)
                random_dir /= np.linalg.norm(random_dir)
                ped.velocity = 0.5 * ped.desired_speed * random_dir
            
            # Update z-coordinate
            ped.z = ped.floor * (self.environment.floor_height + self.environment.floor_thickness)
            
            # Check for stairs if not on ground floor
            if ped.floor > 0 and not ped.on_stairs:
                stair = self.environment.get_stair_from(ped.floor, ped.x, ped.y)
                if stair and abs(ped.x - stair.start[0]) < 2 and abs(ped.y - stair.start[1]) < 2:
                    ped.on_stairs = True
                    ped.current_stair = stair
                    ped.stair_step_idx = 0

            # Handle queueing
            if door_queues is not None:
                handle_door_queueing(ped, door_queues, self.environment)
            if exit_queues is not None:
                handle_exit_queueing(ped, exit_queues, evac_time, self.environment)
            ped.update_color()
            return False
        except Exception as e:
            print(f"[DEBUG][PSO_SFM] Error processing staff {getattr(ped, 'id', None)}: {e}")
            return False

    def calculate_velocity(self, ped, pos, goal, g_best):
        # Ensure goal is a valid numpy array
        if goal is None or not (isinstance(goal, np.ndarray) and goal.shape == (2,)):
            print(f"[DEBUG][PSO_SFM] Invalid goal for ped {getattr(ped, 'id', None)}: {goal}")
            goal = pos + np.random.uniform(-1, 1, 2)
            goal = goal / np.linalg.norm(goal) * 0.5 + pos
        
        goal_direction = goal - pos

        distance = np.linalg.norm(goal_direction)
        if distance > 0:
            desired_direction = goal_direction / distance
        else:
            desired_direction = np.zeros(2)
        v_desired = ped.desired_speed * desired_direction
        f_goal = (v_desired - ped.velocity) / self.tau
        f_repulsion = calculate_social_force(ped, pos, self.pedestrians)
        f_wall = calculate_wall_force(ped, pos, self.environment, self.grid_width, self.grid_height)
        sfm_velocity = f_goal + (f_repulsion + f_wall) / self.MASS
        ped.velocity = sfm_velocity
        # Debug prints
        #print(f"[DEBUG][PSO_SFM] ped {getattr(ped, 'id', None)} pos: {pos}, goal: {goal}, velocity: {ped.velocity}, sfm_velocity: {sfm_velocity}, p_best: {[ped.px_best, ped.py_best]}, g_best: {g_best}")

    def _move_civilian(self, ped, civilians, staff_influence, staff_guidance_goals, priority_group, occupied, occupied_stairs, exit_queues, door_queues, evac_time):
        try:
            if getattr(ped, 'evacuated', False):
                return True  # evacuated
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
                    return False
                except Exception as e:
                    print(f"[DEBUG][PSO_SFM] Error moving civilian on stairs: {e}")
                    return False
            pos = np.array([ped.x, ped.y])
            goal = None

            # Check if civilian is under staff influence based on room containment
            if priority_group is not None and ped in priority_group:
                # Get civilian's current room
                civ_room = self.environment.get_room(ped.x, ped.y, ped.floor)
                if civ_room:
                    # Find staff members in the same room
                    influencing_staff = [
                        s for s, (center, room, floor) in staff_influence.items()
                        if (floor == ped.floor and room == civ_room)
                    ]
                    if influencing_staff:
                        # Find nearest staff member
                        # Choose leader based on staff_priority (lower value = higher priority)
                        leader = min(
                            influencing_staff, 
                            key=lambda s: (getattr(s, 'staff_priority', float('inf')), np.linalg.norm(np.array([s.x, s.y]) - pos))
                        )
                        # Use the staff's guidance_goal as the goal for civilians
                        staff_target = staff_guidance_goals.get(leader, staff_guidance_goals[leader])
                        if isinstance(staff_target, np.ndarray):
                            goal = staff_target
                            ped.following_staff = True
                            # Increase desired speed when following staff
                            ped.desired_speed = min(ped.max_speed, 2.0)
                    else:
                        ped.following_staff = False
                        goal = self.environment.get_nearest_goal(ped, staff_aware=False, is_pso=True)

            # If not under staff influence, use regular SFM or herding
            if not ped.following_staff or goal is None:
                goal = self.environment.get_nearest_goal(ped, staff_aware=False, is_pso=True)
                if goal is None or not (isinstance(goal, np.ndarray) and goal.shape == (2,)):
                    # Herding: align with neighbors in same room
                    neighbor_velocities = []
                    current_room = self.environment.get_room(ped.x, ped.y, ped.floor)
                    if VISIBILITY:
                        for other in civilians:
                            if (other is ped or getattr(other, 'evacuated', False) or 
                                ped.floor != other.floor or 
                                self.environment.get_room(other.x, other.y, other.floor) != current_room):
                                continue
                            neighbor_velocities.append(getattr(other, 'velocity', np.zeros(2)))
                    if neighbor_velocities and VISIBILITY:
                        avg_velocity = np.mean(neighbor_velocities, axis=0)
                        norm = np.linalg.norm(avg_velocity)
                        goal = pos + (avg_velocity / norm if norm > 0 else np.zeros(2))
                    else:
                        random_direction = np.random.uniform(-1, 1, 2)
                        goal = pos + random_direction / np.linalg.norm(random_direction)

            # --- PSO p_best update ---
            if not hasattr(ped, 'px_best'):
                ped.px_best = ped.x
            if not hasattr(ped, 'py_best'):
                ped.py_best = ped.y
            if goal is not None:
                if np.linalg.norm(pos - goal) < np.linalg.norm(np.array([ped.px_best, ped.py_best]) - goal):
                    ped.px_best, ped.py_best = ped.x, ped.y
            # --- PSO g_best update (store for environment) ---
            g_best = self._get_gbest(ped, civilians)
            ped.gx_best, ped.gy_best = g_best[0], g_best[1]

            # Calculate goal-directed force with increased weight
            direction = goal - pos
            distance_to_goal = np.linalg.norm(direction)
            if distance_to_goal > 0:
                desired_direction = direction / distance_to_goal
            else:
                desired_direction = np.zeros(2)

            # Initialize or validate velocity
            if not hasattr(ped, 'velocity') or not isinstance(ped.velocity, np.ndarray):
                ped.velocity = np.zeros(2)

            # Calculate forces with adjusted weights
            v_desired = ped.desired_speed * desired_direction
            f_drive = v_desired - ped.velocity / self.tau
            from .sfm_common import amplify_drive_if_in_queue
            f_drive = amplify_drive_if_in_queue(f_drive, ped)
            f_repulsion = calculate_social_force(ped, pos, self.pedestrians)  # Reduced repulsion
            f_wall = calculate_wall_force(ped, pos, self.environment, self.grid_width, self.grid_height)
            
            # Combine forces with goal-directed behavior having higher priority
            force = f_drive + (f_repulsion + f_wall) / self.MASS
            
            # Update velocity with momentum
            ped.velocity = ped.velocity + force * TIME_STEP  # Add momentum

            # Dampen speed if in queue
            if getattr(ped, '_in_exit_queue', False) or getattr(ped, '_in_door_queue', False):
                ped.velocity *= 0.7  # Damping factor for queueing
            
            speed = np.linalg.norm(ped.velocity)
            if speed > ped.max_speed:
                ped.velocity *= (ped.desired_speed / speed)

            # Update position
            new_x = ped.x + ped.velocity[0] * TIME_STEP
            new_y = ped.y + ped.velocity[1] * TIME_STEP
            new_x = np.clip(new_x, 0.5, self.grid_width - 0.5)
            new_y = np.clip(new_y, 0.5, self.grid_height - 0.5)

            # Track movement history for stuck detection
            if not hasattr(ped, '_pos_history'):
                ped._pos_history = []
            ped._pos_history.append((ped.x, ped.y))
            if len(ped._pos_history) > 10:
                ped._pos_history.pop(0)

            # Check if stuck (hasn't moved significantly in last 10 steps)
            if len(ped._pos_history) == 10:
                dists = [np.linalg.norm(np.array([ped.x, ped.y]) - np.array(pos)) for pos in ped._pos_history]
                if max(dists) < 0.5:
                    # Nudge with a random direction
                    nudge = np.random.uniform(-1, 1, 2)
                    nudge /= np.linalg.norm(nudge)
                    ped.velocity += 0.5 * ped.desired_speed * nudge

            # Update position
            new_x = ped.x + ped.velocity[0] * TIME_STEP
            new_y = ped.y + ped.velocity[1] * TIME_STEP
            new_x = np.clip(new_x, 0.5, self.grid_width - 0.5)
            new_y = np.clip(new_y, 0.5, self.grid_height - 0.5)

            # Check occupancy and update position
            target = (ped.floor, int(round(new_x)), int(round(new_y)))
            present = (ped.floor, int(round(ped.x)), int(round(ped.y)))
            if target not in occupied and not self.environment.is_blocked(ped.x, ped.y, new_x, new_y):
                ped.x, ped.y = new_x, new_y
                occupied.discard(present)
                occupied.add(target)

            # Add small random movement if stuck
            if np.linalg.norm(ped.velocity) < 1e-3:
                ped.velocity = np.random.uniform(-1, 1, 2)
                ped.velocity /= np.linalg.norm(ped.velocity)
                ped.velocity *= ped.desired_speed

            # Update z-coordinate and check for stairs
            ped.z = ped.floor * (self.environment.floor_height + self.environment.floor_thickness)
            if ped.floor > 0 and not ped.on_stairs:
                stair = self.environment.get_stair_from(ped.floor, ped.x, ped.y)
                if stair and abs(ped.x - stair.start[0]) < 2 and abs(ped.y - stair.start[1]) < 2:
                    ped.on_stairs = True
                    ped.current_stair = stair
                    ped.stair_step_idx = 0

            # Handle queuing
            if door_queues is not None:
                handle_door_queueing(ped, door_queues, self.environment)
            if exit_queues is not None:
                handle_exit_queueing(ped, exit_queues, evac_time, self.environment)
            ped.update_color()
            return False

        except Exception as e:
            print(f"[DEBUG][PSO_SFM] Error processing civilian {getattr(ped, 'id', None)}: {e}")
            return False
        
    def _assign_staff_goal(self, staff):
        """Assign a goal to staff based on policy and current state."""
        return self.environment.get_nearest_goal(staff, staff_aware=True, is_pso=True)

    def _get_gbest(self, ped, civilians):
        """Return the g_best for a pedestrian: best p_best in same room, then floor, then global, using environment's evacuation_heuristic."""
        pos = np.array([ped.x, ped.y])
        env = self.environment
        # 1. Try same room
        room = env.get_room(ped.x, ped.y, ped.floor)
        if room:
            room_civilians = [
                other for other in civilians
                if env.get_room(other.x, other.y, other.floor) == room
            ]
            if room_civilians:
                g_best_ped = min(
                    room_civilians,
                    key=lambda other: env.evacuation_heuristic(
                        other,
                        np.array([getattr(other, 'px_best', other.x), getattr(other, 'py_best', other.y)]),
                        exit_room=room,
                        exit_floor=ped.floor
                    )
                )
                return np.array([getattr(g_best_ped, 'px_best', g_best_ped.x), getattr(g_best_ped, 'py_best', g_best_ped.y)])
        # 2. Try same floor
        floor_civilians = [other for other in civilians if other.floor == ped.floor]
        if floor_civilians:
            g_best_ped = min(
                floor_civilians,
                key=lambda other: env.evacuation_heuristic(
                    other,
                    np.array([getattr(other, 'px_best', other.x), getattr(other, 'py_best', other.y)]),
                    exit_room=room,
                    exit_floor=ped.floor
                )
            )
            return np.array([getattr(g_best_ped, 'px_best', g_best_ped.x), getattr(g_best_ped, 'py_best', g_best_ped.y)])
        # 3. Fallback: global
        g_best_ped = min(
            civilians,
            key=lambda other: env.evacuation_heuristic(
                other,
                np.array([getattr(other, 'px_best', other.x), getattr(other, 'py_best', other.y)]),
                exit_room=room,
                exit_floor=ped.floor
            )
        )
        return np.array([getattr(g_best_ped, 'px_best', g_best_ped.x), getattr(g_best_ped, 'py_best', g_best_ped.y)])

