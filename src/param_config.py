# Simulation Space
GRID_WIDTH = 30  # in meters
GRID_LENGTH = 30  # in meters
FLOOR_COUNT = 3
FLOOR_WIDTH = 30   # in meters
FLOOR_LENGTH = 30  # in meters
FLOOR_HEIGHT = 10  # vertical distance between floors
FLOOR_THICKNESS = 0.5 # thickness of each floor slab

# Pedestrian
N_PEDESTRIANS = 120
PEDESTRIAN_SIZE = 0.4
PED_RADIUS = 0.15     # Physical radius in meters
DESIRED_SPEED = 1.3
HELICAL_CONSTANT_B = 2

# Stairs
STAIR_HEIGHT = 0.4  # height of one stair step
STAIR_DEPTH = 0.8   # depth of one stair step
STAIR_WIDTH = 2.2   # width of staircase
STAIR_CAPACITY = int(STAIR_WIDTH // (2 * PED_RADIUS))  # max number of pedestrians side by side on stairs
STAIR_GAP = 0.05    # minimum gap between pedestrians on stairs

#Opacity Settings
FLOOR_OPACITY = 0.7
WALL_OPACITY = 0.25
STAIR_OPACITY = 0.5
DOOR_OPACITY = 0.7

NUM_STAIRS = 4
STAIR_STEP_HEIGHT = 0.2  # Height of each step in meters
# Exit
EXITS = [{'x': 5, 'y': 0}, {'x': 25, 'y': 0}]  # Two exits on ground floor
DOOR_WIDTH = 2
MAX_PASS_LIMIT = int(DOOR_WIDTH // (2 * PED_RADIUS))  # Maximum number of pedestrians that can pass through the door at once

# Simulation Control
MAX_ITERATIONS = 200
TIME_STEP = 1  # seconds

# Rectangle-based queue parameters
QUEUE_RECTANGLE_LENGTH = DOOR_WIDTH * 1.5  # Length of the rectangle (matches door width)
QUEUE_RECTANGLE_WIDTH = 1.6 # Small width to define the evacuation zone

#Visibility Settings
VISIBILITY = True


#SFM
# Constants from Helbing & Molnar (1995)
A = 2       # interaction strength
B = 0.08       # interaction range
k = 120000     # body force constant
kappa = 240000 # sliding friction
tau = 0.5
mass = 80


# PSO-SFM Parameters  
STAFF_RATIO = 0.05  # 5% of pedestrians will be staff/leaders
MIN_STAFF = 15

# Staff Coordination Parameters
COMPLIANCE_RATE_MIN = 0.8  # Increase minimum compliance rate
COMPLIANCE_RATE_MAX = 0.95  # Ensure maximum compliance rate
STAFF_INFLUENCE_RADIUS = 12.0  # Increase influence radius for stronger staff guidance
STAFF_COMMUNICATION_RADIUS = None  # Remove communication radius

# Strategic Positions
STRATEGIC_POSITIONS = {
    0: [(5, 5), (25, 5), (5, 25), (25, 25)],  # Ground floor (near exits)
    1: [(5, 15), (25, 15), (15, 5), (15, 25)],  # First floor (near stairs)
    2: [(5, 15), (25, 15), (15, 5), (15, 25)]   # Second floor (near stairs)
}

# Family/Group Modeling Parameters
FAMILY_ATTRACTION_STRENGTH = 10.0  # Strength of attraction to family/group members
FAMILY_ATTRACTION_RANGE = 20.0      # Max distance for family attraction to apply (meters)

# Panic Mode Parameters
PANIC_RADIUS_MULTIPLIER = 1.5  # Multiplier for radius during panic
PANIC_BODY_FORCE_MULTIPLIER = 2.0  # Multiplier for body force during panic
PANIC_FRICTION_MULTIPLIER = 2.5  # Multiplier for sliding friction during panic

# Toggle for showing visuals (VPython/Plotly)
SHOW_VISUALS = True  # Set to True to enable visualizations

# Staff policy options: 'all_evacuate', 'half_assist_half_leave', 'all_assist', 'assist_mobile_first', 'assist_elderly_first'
STAFF_POLICY = 'all_evacuate'  # Change this to switch policy

STAFF_POLICIES = {
    'all_evacuate': 'All staff evacuate immediately',
    'half_assist_half_leave': 'Half staff assist, half evacuate',
    'all_assist': 'All staff stay and assist',
    'assist_mobile_first': 'Staff assist mobile first, then elderly',
    'assist_elderly_first': 'Staff assist elderly first, then mobile',
    'top_evac': 'Evacuate top floor first, then next lower, etc.',
    'avoid_top': 'Prioritise bottom floors, avoid top (fire in middle)',
    'zone_sweep': 'Sweep zones, starting from bottom left to top right'
}