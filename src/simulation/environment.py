from param_config import FLOOR_HEIGHT, EXITS, FLOOR_THICKNESS
import plotly.graph_objects as go
import numpy as np
import random

class Staircase:
    def __init__(self, start_x, start_y, from_floor, to_floor):
        self.start = (start_x, start_y)
        self.from_floor = from_floor
        self.to_floor = to_floor

class Environment:
    def __init__(self):
        self.exits = EXITS
        self.staircases = self._generate_staircases()
        self.floor_height = FLOOR_HEIGHT
        self.floor_thickness = FLOOR_THICKNESS  # Thickness of each floor
        self.floor_count = 3  # Example: 3 floors
        self.width = 30
        self.length = 30
        self.internal_walls = self._generate_internal_walls()
        self.outer_walls = [
            (0, 0, 30, 30)  # Bottom wall
            ]
        # Precompute optimal door paths for each (room, stairs_room, floor)
        self.room_graph = self._build_room_graph()
        self.optimal_door_paths = self._precompute_optimal_paths()
        self.doors = [{'x': d[0], 'y': d[1]} for d in self.get_doors()]


    def _generate_staircases(self):
        # Two realistic staircases per floor, placed at (5, 5) and (25, 25)
        stairs = []
        positions = [(5, 25), (25, 25)]
        for (x, y) in positions:
            for f in range(2, 0, -1):  # from floor 3 to floor 1
                stairs.append(Staircase(x, y, f, f - 1))
        return stairs

    def _precompute_optimal_paths(self):
        """
        Precompute the optimal door path from every room to every target room for each floor.
        On floor 0, targets are exit rooms; on other floors, targets are stair rooms.
        Returns: dict {(floor, start_room, target_room): [door_x, door_y, ...]}
        """
        from collections import deque
        paths = {}
        rooms = self.internal_walls
        for floor in range(self.floor_count):
            if floor == 0:
                # On ground floor, targets are exit rooms
                target_rooms = set()
                for exit in self.exits:
                    room = self.get_room(exit['x'], exit['y'])
                    if room:
                        target_rooms.add(room)
            else:
                # On other floors, targets are stair rooms
                stairs_on_floor = [s for s in self.staircases if s.from_floor == floor]
                target_rooms = set(self.get_room(s.start[0], s.start[1]) for s in stairs_on_floor)
                target_rooms = {r for r in target_rooms if r is not None}

            for start_room in rooms:
                for target_room in target_rooms:
                    visited = set()
                    queue = deque([(start_room, [])])
                    found = False
                    while queue and not found:
                        room, path = queue.popleft()
                        if room == target_room:
                            paths[(floor, start_room, target_room)] = path
                            found = True
                            break
                        visited.add(room)
                        for neighbor, door_x, door_y in self.room_graph.get(room, []):
                            if neighbor not in visited:
                                queue.append((neighbor, path + [(door_x, door_y)]))
                    if not found:
                        paths[(floor, start_room, target_room)] = []
        return paths

    def update_pedestrian_room(self, ped):
        """
        Update the pedestrian's current_room property based on their (x, y).
        """
        ped.current_room = self.get_room(ped.x, ped.y)

    def get_stair_from(self, floor, x, y):
        """
        Find the nearest staircase on the given floor from the given coordinates.
        If no staircase is found in the same room, return the nearest exit or door.
        """
        def in_rect(rect, x, y):
            x1, y1, x2, y2 = rect
            return x1 <= x <= x2 and y1 <= y <= y2

        # Combine internal and outer walls as rooms
        rooms = self.internal_walls
        containing_room = None
        for room in rooms:
            if in_rect(room, x, y):
                containing_room = room
                break

        if not containing_room:
            return None

        # Find staircases on this floor whose start is inside the same room
        stairs_on_floor = [s for s in self.staircases if s.from_floor == floor and in_rect(containing_room, s.start[0], s.start[1])]
        if not stairs_on_floor:
            return None

        # Return the closest staircase in the room
        return min(stairs_on_floor, key=lambda s: (s.start[0] - x) ** 2 + (s.start[1] - y) ** 2)
    
    def get_nearest_exit(self, x, y):
        return min(self.exits, key=lambda e: (e['x'] - x) ** 2 + (e['y'] - y) ** 2)
    
    def _build_room_graph(self):
        """
        Build a graph where nodes are rooms and edges are doors connecting rooms.
        Returns: dict {room: [(neighbor_room, door_x, door_y), ...]}
        Handles doors that connect more than two rooms (e.g., at corners/junctions).
        """
        rooms = self.internal_walls
        graph = {room: [] for room in rooms}
        for door_x, door_y, _ in self.get_doors():
            # Find all rooms that this door touches
            connected_rooms = []
            for room in rooms:
                x1, y1, x2, y2 = room
                if x1 <= door_x <= x2 and y1 <= door_y <= y2:
                    connected_rooms.append(room)
            # Fully connect all rooms that share this door
            for i in range(len(connected_rooms)):
                for j in range(i + 1, len(connected_rooms)):
                    room1 = connected_rooms[i]
                    room2 = connected_rooms[j]
                    graph[room1].append((room2, door_x, door_y))
                    graph[room2].append((room1, door_x, door_y))
        return graph

    def _find_door_path_to_goal(self, start_room, stairs_room):
        """
        Find the shortest path (sequence of doors) from start_room to stairs_room.
        Returns: list of (door_x, door_y) or empty if no path.
        """
        from collections import deque
        graph = self._build_room_graph()
        visited = set()
        queue = deque([(start_room, [])])
        while queue:
            room, path = queue.popleft()
            if room == stairs_room:
                return path
            visited.add(room)
            for neighbor, door_x, door_y in graph.get(room, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [(door_x, door_y)]))
        return []

    def evacuation_heuristic(self, ped, goal_pos, exit_room=None, exit_floor=0):
        """
        Heuristic for PSO/SFM:
        - Uses 3D distance (x, y, z)
        - Penalizes being in a different room from the exit
        - Penalizes if there is no door path between rooms (blocked)
        - Adds a congestion penalty for crowded rooms
        """
        # 3D distance
        ped_z = getattr(ped, 'z', ped.floor * (self.floor_height + self.floor_thickness))
        if len(goal_pos) == 2:
            goal_z = exit_floor * (self.floor_height + self.floor_thickness)
            goal_xyz = np.array([goal_pos[0], goal_pos[1], goal_z])
        else:
            goal_xyz = np.array(goal_pos)
        ped_xyz = np.array([ped.x, ped.y, ped_z])
        dist = np.linalg.norm(goal_xyz - ped_xyz)
        penalty = 0
        ped_room = getattr(ped, 'current_room', None)
        """# Penalize if not on exit floor
        if hasattr(ped, 'floor') and ped.floor != exit_floor:
            penalty += 30 * abs(ped.floor - exit_floor)
        # Penalize if not in exit room
        if exit_room is not None and ped_room != exit_room:
            penalty += 100
            # Check if there is a door path between rooms
            if (ped.floor, ped_room, exit_room) in self.optimal_door_paths:
                path = self.optimal_door_paths[(ped.floor, ped_room, exit_room)]
                if not path:
                    penalty += 200  # No path, heavily penalize
            else:
                penalty += 200  # No path, heavily penalize"""
        # Congestion penalty: number of pedestrians in the same room
        congestion = 0
        if hasattr(self, 'pedestrians') and ped_room is not None:
            congestion = sum(1 for other in self.pedestrians if getattr(other, 'current_room', None) == ped_room and not getattr(other, 'evacuated', False))
            penalty += 2 * congestion
        return dist + penalty

    def get_nearest_goal(self, ped, staff_aware=False, is_pso=False):
        if not hasattr(ped, 'current_room') or ped.current_room != self.get_room(ped.x, ped.y):
            self.update_pedestrian_room(ped)

        current_room = ped.current_room
        pos = np.array([ped.x, ped.y])

        def door_goal_fallback():
            door = self.get_nearest_door(ped.x, ped.y, floor=ped.floor, staff_aware=staff_aware)
            return np.array([door[0], door[1]]) if door else pos

        def get_exit_and_room():
            nearest_exit = self.get_nearest_exit(ped.x, ped.y)
            exit_room = self.get_room(nearest_exit['x'], nearest_exit['y'])
            return nearest_exit, exit_room

        def get_stair_and_room():
            stairs_on_floor = [s for s in self.staircases if s.from_floor == ped.floor]
            nearest_stair = min(stairs_on_floor, key=lambda s: (s.start[0] - ped.x) ** 2 + (s.start[1] - ped.y) ** 2) if stairs_on_floor else None
            stairs_room = self.get_room(nearest_stair.start[0], nearest_stair.start[1]) if nearest_stair else None
            return nearest_stair, stairs_room

        # --- Staff-aware logic ---
        if staff_aware:
            """# TEMP Immobile staff - give random point as goal
            staff_goal = np.random.uniform(0, self.width, 2)
            staff_goal[1] += self.width / 2  # Offset Y to be within the grid
            staff_goal = np.clip(staff_goal, 0, [self.width, self.length])
            return staff_goal"""
        
            if ped.floor > 0:
                nearest_stair, stairs_room = get_stair_and_room()
                if current_room and stairs_room and current_room != stairs_room:
                    path = self.optimal_door_paths.get((ped.floor, current_room, stairs_room), [])
                    staff_goal = np.array(path[0]) if path else door_goal_fallback()
                else:
                    staff_goal = np.array(nearest_stair.start) if nearest_stair else door_goal_fallback()
                exit_room = stairs_room
                exit_floor = ped.floor
            else:
                nearest_exit, exit_room = get_exit_and_room()
                if current_room == exit_room:
                    staff_goal = np.array([nearest_exit['x'], nearest_exit['y']])
                else:
                    path = self.optimal_door_paths.get((ped.floor, current_room, exit_room), [])
                    staff_goal = np.array(path[0]) if path else door_goal_fallback()
                exit_floor = 0

            room_peds = [other for other in getattr(self, 'pedestrians', []) if getattr(other, 'current_room', None) == current_room]
            if room_peds:
                g_best_ped = min(
                    room_peds,
                    key=lambda other: self.evacuation_heuristic(other, staff_goal, exit_room=exit_room, exit_floor=exit_floor)
                )
                g_best = np.array([getattr(g_best_ped, 'px_best', g_best_ped.x), getattr(g_best_ped, 'py_best', g_best_ped.y)])
                return 0.7 * staff_goal + 0.3 * g_best
            else:
                return staff_goal

        # --- PSO-based goal selection ---
        if is_pso:
            room_peds = [other for other in getattr(self, 'pedestrians', []) if getattr(other, 'current_room', None) == current_room]
            if ped.floor == 0:
                nearest_exit, exit_room = get_exit_and_room()
                if current_room == exit_room:
                    goal_pos = np.array([nearest_exit['x'], nearest_exit['y']])
                else:
                    nearest_door = self.get_nearest_door(ped.x, ped.y, floor=ped.floor, staff_aware=staff_aware)
                    goal_pos = np.array([nearest_door[0], nearest_door[1]]) if nearest_door else pos
                exit_floor = 0
            else:
                nearest_stair, stairs_room = get_stair_and_room()
                if current_room == stairs_room and nearest_stair:
                    goal_pos = np.array(nearest_stair.start)
                elif nearest_stair:
                    nearest_door = self.get_nearest_door(ped.x, ped.y, floor=ped.floor, staff_aware=staff_aware)
                    goal_pos = np.array([nearest_door[0], nearest_door[1]]) if nearest_door else pos
                else:
                    goal_pos = pos
                exit_room = stairs_room
                exit_floor = ped.floor

            if room_peds:
                g_best_ped = min(
                    room_peds,
                    key=lambda other: self.evacuation_heuristic(other, goal_pos, exit_room=exit_room, exit_floor=exit_floor)
                )
                g_best = np.array([getattr(g_best_ped, 'px_best', g_best_ped.x), getattr(g_best_ped, 'py_best', g_best_ped.y)])
                return g_best
            return goal_pos

        # --- Fallback: normal goal logic ---
        if ped.floor == 0:
            nearest_exit, exit_room = get_exit_and_room()
            goal = np.array([nearest_exit['x'], nearest_exit['y']]) if current_room == exit_room else door_goal_fallback()
        else:
            target = self.get_stair_from(ped.floor, ped.x, ped.y)
            goal = np.array(target.start) if target else door_goal_fallback()

        return goal

    def get_exit_locations(self):
        return [(e['x'], e['y']) for e in self.exits]
    
    def get_nearest_stair(self, floor, x, y):
        stairs_on_floor = [s for s in self.staircases if s.from_floor == floor]
        if not stairs_on_floor:
            return None
        return min(stairs_on_floor, key=lambda s: (s.start[0] - x) ** 2 + (s.start[1] - y) ** 2)

    def _generate_internal_walls(self):
        # Define rooms as rectangles: (x1, y1, x2, y2)
        # Adding more complex internal walls and dividers
        return [
            (0, 0, 10, 15),    # Room 1
            (10, 0, 20, 15),   # Room 2
            (20, 0, 30, 15),   # Room 3
            (0, 15, 15, 30),   # Room 4
            (15, 15, 30, 30)  # Room 5
        ]

    def get_walls(self):
        # Combine outer and internal walls
        return self.outer_walls + self.internal_walls

    def get_doors(self):
        # Adding more doors for the complex layout with direction
        return [
            (10, 7, "vertical"),  # Door between Room 1 and Room 2
            (20, 7, "vertical"),  # Door between Room 2 and Room 3
            (15, 20, "vertical"), # Door between Room 4 and Room 5
            (7, 15, "horizontal"),  # Rotated door between Room 1 and Room 4
            (22, 15, "horizontal"), # Rotated door between Room 3 and Room 5
        ]

    def get_room(self, x, y, floor=None):
        """
        Determine which room a coordinate belongs to.
        Returns the room as a tuple (x1, y1, x2, y2) or None if outside all rooms.
        """
        for room in self.internal_walls:
            x1, y1, x2, y2 = room
            if x1 <= x <= x2 and y1 <= y <= y2:
                return room
        return None

    def is_blocked(self, x_before, y_before, x_after, y_after):
        """
        Check if movement from (x_before, y_before) to (x_after, y_after) is blocked.
        Blocked if transitioning between rooms without using a door.
        """
        room_before = self.get_room(x_before, y_before)
        room_after = self.get_room(x_after, y_after)

        # If both coordinates are in the same room, movement is not blocked
        if room_before == room_after:
            return False

        # If transitioning between rooms, check if movement passes through a door
        for door_x, door_y, _ in self.get_doors():
            if min(x_before, x_after) <= door_x <= max(x_before, x_after) and \
               min(y_before, y_after) <= door_y <= max(y_before, y_after):
                return False  # Movement passes through a door

        return True  # Movement is blocked if transitioning between rooms without a door

    def get_num_floors(self):
        return self.floor_count

    def get_stair_end(self, stair):
        """
        Returns the (x, y, z) end position of the given staircase.
        """
        x0, y0 = stair.start
        floor_height = self.floor_height
        if hasattr(stair, 'direction') and stair.direction == 'up':
            if x0 < self.width/2:
                x1 = x0
                y1 = y0 - 3
            else:
                x1 = x0
                y1 = y0 + 3
        else:  # Default or 'down'
            if x0 < self.width/2:
                x1 = x0
                y1 = y0 + 3
            else:
                x1 = x0
                y1 = y0 - 3
        z1 = stair.to_floor * floor_height
        return (x1, y1, z1)

    def get_stairs_on_floor(self, floor):
        """
        Retrieve all staircases on a specific floor.
        """
        return [stair for stair in self.staircases if stair.from_floor == floor]

    def get_nearest_door(self, x, y, floor, staff_aware=False):
        """
        Find the nearest door on the given floor from the given coordinates.
        If staff_aware is True, find the nearest door to the nearest stair on that floor.
        Otherwise, return the nearest door to (x, y), but with a small probability, return a random other door.
        """
        doors_on_floor = self.get_doors()
        stairs_on_floor = [s for s in self.staircases if s.from_floor == floor]
        if not doors_on_floor:
            return None

        if staff_aware and stairs_on_floor:
            # Find the nearest stair to (x, y)
            nearest_stair = min(stairs_on_floor, key=lambda s: (s.start[0] - x) ** 2 + (s.start[1] - y) ** 2)
            # Find the door nearest to this stair
            return min(doors_on_floor, key=lambda d: (d[0] - nearest_stair.start[0]) ** 2 + (d[1] - nearest_stair.start[1]) ** 2)
        else:
            # Mostly return the nearest door, but sometimes return another random door
            nearest = min(doors_on_floor, key=lambda d: (d[0] - x) ** 2 + (d[1] - y) ** 2)
            other_doors = [d for d in doors_on_floor if d != nearest]
            # 90% nearest, 10% random other
            if other_doors and random.random() < 0.1:
                return random.choice(other_doors)
            else:
                return nearest

    def get_rooms_near_door(self, door_x, door_y):
        """
        Determine the rooms connected by a door.
        Returns a tuple of two rooms (room1, room2) or None if the door does not connect two rooms.
        """
        connected_rooms = []
        for room in self.internal_walls:
            x1, y1, x2, y2 = room
            if x1 <= door_x <= x2 and y1 <= door_y <= y2:
                connected_rooms.append(room)

        if len(connected_rooms) == 2:
            return tuple(connected_rooms)
        return None

    def get_stair_to(self, from_floor, to_floor):
        """
        Get a staircase that connects from from_floor to to_floor.
        """
        for stair in self.staircases:
            if stair.from_floor == from_floor and stair.to_floor == to_floor:
                return stair
        return None
    
    def get_room_bounds(self, room):
        """
        Get the bounds of a room as (x_min, y_min, x_max, y_max).
        Args:
            room: A tuple (x1, y1, x2, y2) representing a room's corners
        Returns:
            tuple: (x_min, y_min, x_max, y_max) or None if room is invalid
        """
        if not room or not isinstance(room, tuple) or len(room) != 4:
            return None
        
        x1, y1, x2, y2 = room
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def get_room_center(self, room):
        """
        Get the center coordinates of a room.
        Args:
            room: A tuple (x1, y1, x2, y2) representing a room's corners
        Returns:
            tuple: (center_x, center_y) or None if room is invalid
        """
        bounds = self.get_room_bounds(room)
        if not bounds:
            return None
        
        x_min, y_min, x_max, y_max = bounds
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)
