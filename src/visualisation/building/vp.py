from vpython import box, sphere, cylinder, vector, color, scene, rate, compound
import random

# Scene setup
scene.title = "2-Floor Building with Walking Stick Humans"
scene.background = color.white
scene.center = vector(0, 10, 0)
scene.forward = vector(-1, -0.5, -1)

# Parameters
floor_width = 20
floor_length = 20
floor_height = 0.5
floor_gap = 6

# Colors
purple = color.purple
stair_color = vector(0.8, 0.2, 0.8)

# Create Floors
floor1 = box(pos=vector(0, 0, 0), size=vector(floor_width, floor_height, floor_length), color=purple)
floor2 = box(pos=vector(0, floor_gap, 0), size=vector(floor_width, floor_height, floor_length), color=purple)

# Doors
door_width = 2
door_height = 3
door_thickness = 0.1
door1 = box(pos=vector(0, door_height/2, -floor_length/2), size=vector(door_width, door_height, door_thickness), color=color.red)
door2 = box(pos=vector(0, floor_gap + door_height/2, floor_length/2), size=vector(door_width, door_height, door_thickness), color=color.red)
entrance = box(pos=vector(floor_width/2, door_height/2, 0), size=vector(door_thickness, door_height, door_width), color=color.green)

# Staircases
stair_width = 3
stair_thickness = 0.4
stair_length = floor_gap * 2.2
stair1_pos = vector(-floor_width/4, floor_gap/2, 0)
stair2_pos = vector(floor_width/4, floor_gap/2, 0)

stair1 = box(pos=stair1_pos, size=vector(stair_width, stair_thickness, stair_length), color=stair_color)
stair1.rotate(angle=0.5, axis=vector(1, 0, 0), origin=stair1.pos)

stair2 = box(pos=stair2_pos, size=vector(stair_width, stair_thickness, stair_length), color=stair_color)
stair2.rotate(angle=0.5, axis=vector(1, 0, 0), origin=stair2.pos)

# Collect all parts for rotation
building_parts = [floor1, floor2, door1, door2, entrance, stair1, stair2]

# Stick figures
num_humans = 4
stick_figures = []
floor_levels = {1: floor_height + 1, 2: floor_gap + floor_height + 1}

for _ in range(num_humans):
    x = random.uniform(-floor_width/2 + 1, floor_width/2 - 1)
    z = random.uniform(-floor_length/2 + 1, floor_length/2 - 1)
    floor_num = random.choice([1, 2])
    y_base = floor_levels[floor_num]

    # Create parts
    head = sphere(pos=vector(0, 1.6, 0), radius=0.4, color=color.blue)
    body = cylinder(pos=vector(0, 0.5, 0), axis=vector(0, 1.0, 0), radius=0.2, color=color.blue)
    arm_left = cylinder(pos=vector(-0.3, 1.2, 0), axis=vector(-0.5, -0.5, 0), radius=0.1, color=color.blue)
    arm_right = cylinder(pos=vector(0.3, 1.2, 0), axis=vector(0.5, -0.5, 0), radius=0.1, color=color.blue)
    leg_left = cylinder(pos=vector(-0.2, 0, 0), axis=vector(0, -0.8, 0), radius=0.1, color=color.blue)
    leg_right = cylinder(pos=vector(0.2, 0, 0), axis=vector(0, -0.8, 0), radius=0.1, color=color.blue)

    # Combine and position
    stick = compound([head, body, arm_left, arm_right, leg_left, leg_right])
    stick.pos = vector(x, y_base, z)
    stick.floor_num = floor_num  # track which floor it's on
    stick_figures.append(stick)
    building_parts.append(stick)

# Rotation
rotation_angle = 0.1
rotation_origin = vector(0, floor_gap/2, 0)

def rotate_building(axis, direction):
    for part in building_parts:
        part.rotate(angle=direction * rotation_angle, axis=axis, origin=rotation_origin)

def key_input(evt):
    key = evt.key
    if key == 'left':
        rotate_building(vector(0,1,0), -1)
    elif key == 'right':
        rotate_building(vector(0,1,0), 1)
    elif key == 'up':
        rotate_building(vector(1,0,0), -1)
    elif key == 'down':
        rotate_building(vector(1,0,0), 1)

scene.bind('keydown', key_input)

# Movement loop
while True:
    rate(20)
    for stick in stick_figures:
        dx = random.uniform(-0.2, 0.2)
        dz = random.uniform(-0.2, 0.2)
        new_x = stick.pos.x + dx
        new_z = stick.pos.z + dz

        # Bounds
        new_x = max(-floor_width/2 + 1, min(floor_width/2 - 1, new_x))
        new_z = max(-floor_length/2 + 1, min(floor_length/2 - 1, new_z))

        # Change floor if near stairs
        for stair_pos, target_floor in [(stair1_pos, 2), (stair2_pos, 2), (stair1_pos, 1), (stair2_pos, 1)]:
            if abs(stick.pos.x - stair_pos.x) < 2 and abs(stick.pos.z - stair_pos.z) < 2:
                if random.random() < 0.01:
                    stick.floor_num = 3 - stick.floor_num  # toggle floor
                    stick.pos.y = floor_levels[stick.floor_num]

        stick.pos = vector(new_x, stick.pos.y, new_z)
