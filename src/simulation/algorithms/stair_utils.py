import numpy as np
from param_config import STAIR_HEIGHT, STAIR_DEPTH, STAIR_WIDTH, STAIR_CAPACITY, PED_RADIUS

def move_pedestrian_on_stairs(ped, pedestrians, occupied_stairs, environment, floor_height, floor_thickness):
    """
    Move a pedestrian one step down the stairs, updating their position and state.
    Args:
        ped: The pedestrian object (must have .current_stair, .stair_step_idx, .x, .y, .z, .floor, .on_stairs, etc.)
        pedestrians: List of all pedestrians for checking step occupancy
        occupied_stairs: Set of (stair_id, stair_step_idx) tuples representing occupied stair steps
        environment: The environment object (must have .floor_height, .floor_thickness)
        floor_height: Height of a floor
        floor_thickness: Thickness of a floor
    Returns:
        None (modifies ped in place)
    """
    try:
        stair = ped.current_stair
        step_idx = getattr(ped, 'stair_step_idx', 0) + 1

        x0, y0 = stair.start
        # Account for pedestrian height in z calculation
        ped_height = getattr(ped, 'height', 1.75)  # default height 1.75m
        z1 = stair.from_floor * (floor_height + floor_thickness) + ped_height / 2
        z2 = stair.to_floor * (floor_height + floor_thickness) + ped_height / 2
        dz = z2 - z1

        num_steps = int(abs(z2 - z1) / STAIR_HEIGHT)

        # Calculate stair path
        stair_angle_deg = 30 if x0 > 2 and y0 > 2 else 330
        stair_length = abs(dz) / np.sin(np.deg2rad(stair_angle_deg if stair_angle_deg != 0 else 1))
        dx = 0
        dy = np.sign(dz) * np.sqrt(max(stair_length**2 - dz**2, 0))
        end_x = x0 + dx
        end_y = y0 + dy

        # Check current step occupancy
        current_step_occupants = [(p.x, p.y) for p in pedestrians 
                                if hasattr(p, 'current_stair') and p.current_stair == stair 
                                and p.stair_step_idx == step_idx 
                                and p != ped]
        
        # Determine position on step (left or right)
        offset = STAIR_WIDTH/4  # Quarter of stair width for side position
        if len(current_step_occupants) < STAIR_CAPACITY:
            # Calculate base position on step
            base_x = x0 + step_idx * (end_x - x0) / num_steps
            base_y = y0 + step_idx * (end_y - y0) / num_steps
            
            # If step empty, take left side, else take right side
            if not current_step_occupants:
                # Left side
                if x0 < 15:  # Left staircase
                    ped.x = base_x - offset
                else:  # Right staircase
                    ped.x = base_x + offset
            else:
                # Take opposite side from existing occupant
                other_x, _ = current_step_occupants[0]
                if other_x < base_x:
                    ped.x = base_x + offset  # Take right side
                else:
                    ped.x = base_x - offset  # Take left side
            
            ped.y = base_y
            ped.z = z1 + step_idx * (z2 - z1) / num_steps
            ped.stair_step_idx = step_idx

            next_step_idx = ped.stair_step_idx + 1
            target_stair = (id(stair), next_step_idx)
            if target_stair not in occupied_stairs:
                ped.stair_step_idx = next_step_idx
                occupied_stairs.add(target_stair)

        # Check if reached end of stairs
        if step_idx >= num_steps:
            ped.floor = stair.to_floor
            ped.z = z2
            ped.on_stairs = False
            ped.current_stair = None
            ped.stair_step_idx = 0
            
            # Maintain side offset at stair exit
            if x0 < 15:  # Left staircase
                ped.x = x0 - offset
            else:  # Right staircase
                ped.x = x0 + offset
            ped.y = end_y

    except Exception as e:
        print(f"[DEBUG][STAIR_UTILS] Error moving pedestrian on stairs: {e}")
