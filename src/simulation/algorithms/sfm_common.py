import numpy as np
from param_config import PED_RADIUS, A, B, k, kappa, mass, FAMILY_ATTRACTION_STRENGTH, FAMILY_ATTRACTION_RANGE, PANIC_RADIUS_MULTIPLIER, PANIC_BODY_FORCE_MULTIPLIER, PANIC_FRICTION_MULTIPLIER, QUEUE_RECTANGLE_LENGTH, QUEUE_RECTANGLE_WIDTH, MAX_PASS_LIMIT

def calculate_family_force(ped, pos, all_peds):
    """
    Attraction force towards family/group members.
    """
    if not hasattr(ped, 'crowd_id') or ped.crowd_id is None:
        return np.zeros(2)
    f_family = np.zeros(2)
    count = 0
    for other in all_peds:
        if other is ped or getattr(other, 'evacuated', False):
            continue
        if getattr(other, 'crowd_id', None) == ped.crowd_id and ped.floor == other.floor:
            other_pos = np.array([other.x, other.y])
            diff = other_pos - pos
            dist = np.linalg.norm(diff)
            if 0 < dist < FAMILY_ATTRACTION_RANGE:
                f_family += FAMILY_ATTRACTION_STRENGTH * diff / (dist + 1e-6)
                count += 1
    if count > 0:
        f_family /= count
    return f_family

def calculate_social_force(ped, pos, other_peds):
    """Calculate repulsive force between pedestrians (SFM) + family attraction + panic mode adjustments. If in exit/door queue, reduce repulsion."""
    # Set panic level based on following staff
    ped.panicked = not ped.following_staff if hasattr(ped, 'following_staff') else True
    f_repulsion = np.zeros(2)
    panic_multiplier = getattr(ped, 'panic_level', 1.0)
    dynamic_radius = PED_RADIUS * panic_multiplier * PANIC_RADIUS_MULTIPLIER
    dynamic_k = k * panic_multiplier * PANIC_BODY_FORCE_MULTIPLIER
    dynamic_kappa = kappa * panic_multiplier * PANIC_FRICTION_MULTIPLIER

    # If in exit/door queue, reduce repulsion
    in_queue = getattr(ped, '_in_exit_queue', False) or getattr(ped, '_in_door_queue', False)
    repulsion_scale = 0.2 if in_queue else 1.0

    for other in other_peds:
        if other is ped or getattr(other, 'evacuated', False):
            continue

        # Skip if on different floors and not on stairs close by
        if ped.floor != other.floor:
            if not (getattr(ped, 'on_stairs', False) or getattr(other, 'on_stairs', False)):
                continue
            pos3d = np.array([pos[0], pos[1], ped.z])
            other_pos3d = np.array([other.x, other.y, other.z])
            if np.linalg.norm(pos3d - other_pos3d) > 2.0:
                continue

        r_ij = 2 * dynamic_radius
        dij_vec = pos - np.array([other.x, other.y])
        dij = np.linalg.norm(dij_vec)
        if dij == 0:
            continue
        n_ij = dij_vec / dij
        t_ij = np.array([-n_ij[1], n_ij[0]])
        delta_v_t = np.dot(other.velocity - ped.velocity, t_ij)
        g = max(0, r_ij - dij)
        f_ij = A * np.exp((r_ij - dij) / B) * n_ij
        # Clip exponent to avoid overflow
        exp_term = np.exp(np.clip((r_ij - dij) / B, -100, 100))
        f_ij = A * exp_term * n_ij
        # Clip multiplication results to avoid overflow
        k_term = np.clip(dynamic_k * g, -1e6, 1e6)
        kappa_term = np.clip(dynamic_kappa * g * delta_v_t, -1e6, 1e6)
        f_ij += k_term * n_ij
        f_ij += kappa_term * t_ij
        f_repulsion += repulsion_scale * f_ij

    # Add family/group attraction
    #f_repulsion += repulsion_scale * calculate_family_force(ped, pos, other_peds)
    return f_repulsion

def calculate_wall_force(ped, pos, environment, grid_width, grid_height):
    """Calculate wall repulsion force (SFM) with panic mode adjustments"""
    f_wall = np.zeros(2)
    panic_multiplier = getattr(ped, 'panic_level', 1.0)
    dynamic_radius = PED_RADIUS * panic_multiplier * PANIC_RADIUS_MULTIPLIER
    dynamic_k = k * panic_multiplier * PANIC_BODY_FORCE_MULTIPLIER
    dynamic_kappa = kappa * panic_multiplier * PANIC_FRICTION_MULTIPLIER

    walls = environment.get_walls()
    for wall in walls:
        try:
            if hasattr(wall, 'start') and hasattr(wall, 'end'):
                start = np.array(wall.start)
                end = np.array(wall.end)
            elif isinstance(wall, (tuple, list)) and len(wall) == 2:
                start = np.array(wall[0])
                end = np.array(wall[1])
            else:
                continue
            wall_vec = end - start
            wall_len = np.linalg.norm(wall_vec)
            if wall_len == 0:
                continue
            wall_dir = wall_vec / wall_len
            rel_pos = pos - start
            proj_length = np.dot(rel_pos, wall_dir)
            proj_point = start + np.clip(proj_length, 0, wall_len) * wall_dir
            d_vec = pos - proj_point
            d = np.linalg.norm(d_vec)
            if d == 0:
                continue
            n_iw = d_vec / d
            g = max(0, dynamic_radius - d)
            f_wall += (A * np.exp((dynamic_radius - d) / B) + dynamic_k * g) * n_iw
            f_wall += dynamic_kappa * g * np.dot(ped.velocity, wall_dir) * wall_dir
        except Exception as wall_ex:
            print(f"[DEBUG][SFM_COMMON] Wall force error: {wall_ex}")
    return f_wall

def process_exit_queues(exit_queues, evac_time):
    """Robust: logical queue, nudge stuck, allow next to advance, handle just-outside agents."""
    evacuated_peds = 0
    for ex_pos, queue in exit_queues.items():
        # Sort by arrival time, then id for determinism
        queue = sorted(queue, key=lambda tup: (tup[1], id(tup[0])))
        allowed = []
        waiting = []
        allowed_count = 0
        door_start = np.array([ex_pos[0] - QUEUE_RECTANGLE_LENGTH / 2, ex_pos[1]])
        door_end = np.array([ex_pos[0] + QUEUE_RECTANGLE_LENGTH / 2, ex_pos[1] + QUEUE_RECTANGLE_WIDTH])
        # Logical queue: first max_pass_limit non-evacuated are allowed to move (even if not in rectangle)
        logical_queue = [item for item in queue if not getattr(item[0], 'evacuated', False)]
        for idx, (ped, arr_time) in enumerate(logical_queue):
            ped_pos = np.array([ped.x, ped.y])
            in_rect = (door_start[0] <= ped_pos[0] <= door_end[0] and door_start[1] <= ped_pos[1] <= door_end[1])
            
            if hasattr(ped, 'update_color'):
                ped.update_color()
            # Allow first max_pass_limit to move toward exit (even if not in rectangle)
            if allowed_count < MAX_PASS_LIMIT:
                if in_rect:
                    allowed.append((ped, arr_time))
                    allowed_count += 1
                else:
                    # Nudge or set velocity toward rectangle center
                    center = np.array([(door_start[0] + door_end[0]) / 2, (door_start[1] + door_end[1]) / 2])
                    direction = center - ped_pos
                    norm = np.linalg.norm(direction)
                    """if norm > 0:
                        ped.velocity = (direction / norm) * (getattr(ped, 'desired_speed', np.ones(2)))
                    else:
                        ped.velocity = np.ones(2)"""
            else:
                waiting.append((ped, arr_time))
        # Evacuate those in rectangle and allowed
        for ped, _ in allowed:
            if not getattr(ped, 'evacuated', False):
                ped.mark_evacuated(evac_time)
                evacuated_peds += 1
                ped._in_exit_queue = False
                # Nudge out of rectangle
                ped.y += QUEUE_RECTANGLE_WIDTH + 0.1
                #ped.velocity = ped.desired_speed if hasattr(ped, 'desired_speed') else np.zeros(2)
        # For those waiting, freeze
        for ped, _ in waiting:
            if not getattr(ped, 'evacuated', False):
                
                #ped.velocity = ped.velocity * 0.5  # Reduce speed
                ped._in_exit_queue = True
        exit_queues[ex_pos] = [item for item in queue if not getattr(item[0], 'evacuated', False)]
    return evacuated_peds

def staff_nearby_in_queue(ped, queue, radius=3.0):
    """Return True if a staff member is within a given radius of the pedestrian in the exit queue."""
    for other, _ in queue:
        if getattr(other, 'is_staff', False) and ped.floor == other.floor:
            dist = np.linalg.norm(np.array([ped.x, ped.y]) - np.array([other.x, other.y]))
            if dist < radius:
                return True
    return False

def handle_exit_queueing(ped, exit_queues, evac_time, environment):
    """
    Handles exit queueing logic for a pedestrian using rectangle-based zones.
    Returns True if the pedestrian is in the exit queue, False otherwise.
    """
    if ped.floor != 0 or getattr(ped, 'on_stairs', False):
        ped._in_exit_queue = False
        return False

    exit = environment.get_nearest_exit(ped.x, ped.y)
    ex_pos = (exit['x'], exit['y'])
    queue_peds = [item[0] for item in exit_queues[ex_pos]]

    if hasattr(ped, 'update_color'):
        ped.update_color()
    # --- END COLOUR CODING LOGIC ---

    # Define rectangle zone for evacuation
    door_start = np.array([ex_pos[0] - QUEUE_RECTANGLE_LENGTH / 2, ex_pos[1]])
    door_end = np.array([ex_pos[0] + QUEUE_RECTANGLE_LENGTH / 2, ex_pos[1] + QUEUE_RECTANGLE_WIDTH])

    if (door_start[0] <= ped.x <= door_end[0] and
        door_start[1] <= ped.y <= door_end[1] and
        not getattr(ped, 'evacuated', False)):
        if not hasattr(ped, '_in_exit_queue') or not ped._in_exit_queue:
            nearby_in_queue = [qped for qped in queue_peds
                               if door_start[0] <= qped.x <= door_end[0] and
                               door_start[1] <= qped.y <= door_end[1]]
            if nearby_in_queue or len(queue_peds) > 0:
                exit_queues[ex_pos].append((ped, evac_time))
                ped._in_exit_queue = True
                #ped.velocity = np.zeros(2)
            else:
                exit_queues[ex_pos].append((ped, evac_time))
                ped._in_exit_queue = True
        """else:
            ped.velocity = np.zeros(2)"""
        return True
    else:
        ped._in_exit_queue = False
        return False

def process_door_queues(door_queues, environment):
    """Robust: logical queue, nudge stuck, allow next to advance, handle just-outside agents. Now per floor and only for agents on the correct floor."""
    for door_pos, queue in door_queues.items():
        floor, door_x, door_y = door_pos
        # Only process agents on the correct floor
        queue = [item for item in queue if getattr(item[0], 'floor', None) == floor and not getattr(item[0], 'evacuated', False)]
        queue = sorted(queue, key=lambda tup: (tup[1], id(tup[0])))
        door_start = np.array([door_x - QUEUE_RECTANGLE_LENGTH / 2, door_y])
        door_end = np.array([door_x + QUEUE_RECTANGLE_LENGTH / 2, door_y + QUEUE_RECTANGLE_WIDTH])
        allowed = []
        waiting = []
        allowed_count = 0
        for idx, (ped, arr_time) in enumerate(queue):
            ped_pos = np.array([ped.x, ped.y])
            in_rect = (door_start[0] <= ped_pos[0] <= door_end[0] and door_start[1] <= ped_pos[1] <= door_end[1])
            if hasattr(ped, 'update_color'):
                ped.update_color()
            if allowed_count < MAX_PASS_LIMIT:
                if in_rect:
                    allowed.append((ped, arr_time))
                    allowed_count += 1
                else:
                    center = np.array([(door_start[0] + door_end[0]) / 2, (door_start[1] + door_end[1]) / 2])
                    direction = center - ped_pos
                    norm = np.linalg.norm(direction)
            else:
                waiting.append((ped, arr_time))
        for ped, _ in allowed:
            ped._in_door_queue = False
            ped.y += QUEUE_RECTANGLE_WIDTH + 0.1
        for ped, _ in waiting:
            ped._in_door_queue = True
        # Only keep agents on the correct floor and still in the queue
        door_queues[door_pos] = [item for item in queue if getattr(item[0], 'floor', None) == floor and getattr(item[0], '_in_door_queue', False)]

def handle_door_queueing(ped, door_queues, environment):
    """
    Handles door queueing logic for a pedestrian at an internal door, per floor.
    Returns True if the pedestrian is in the door queue, False otherwise.
    """
    for door_x, door_y, _ in environment.get_doors():
        door_start = np.array([door_x - QUEUE_RECTANGLE_LENGTH / 2, door_y])
        door_end = np.array([door_x + QUEUE_RECTANGLE_LENGTH / 2, door_y + QUEUE_RECTANGLE_WIDTH])
        if (door_start[0] <= ped.x <= door_end[0] and door_start[1] <= ped.y <= door_end[1]):
            door_pos = (ped.floor, door_x, door_y)
            queue_peds = [item[0] for item in door_queues.get(door_pos, [])]
            if not hasattr(ped, '_in_door_queue') or not ped._in_door_queue:
                if queue_peds:
                    door_queues.setdefault(door_pos, []).append((ped, getattr(ped, 'arr_time', 0)))
                    ped._in_door_queue = True
                else:
                    door_queues.setdefault(door_pos, []).append((ped, getattr(ped, 'arr_time', 0)))
                    ped._in_door_queue = True
            return True
    ped._in_door_queue = False
    return False

def amplify_drive_if_in_queue(f_drive, ped, factor=2.0):
    """If agent is in exit/door queue, amplify driving force to simulate pushing/panic."""
    if getattr(ped, '_in_exit_queue', False) or getattr(ped, '_in_door_queue', False):
        return f_drive * factor
    return f_drive
