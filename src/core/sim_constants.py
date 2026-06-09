# -----------------------------------------------------
# Simulation Constants
# -----------------------------------------------------

# Simulation steps
MAX_TIME = 60
ACTION_REPEAT = 10

# Simulation timing
NUM_BULLET_SOLVER_ITERATIONS = 30
SIMULATION_TIME_STEP = 0.001

# Camera
RENDER_HEIGHT = 1080
RENDER_WIDTH = 1920
CAMERA_DISTANCE = 2.5
CAMERA_YAW = 0
CAMERA_PITCH = -30

BOX = [
    {'position': [0, 5, 0.25], 'half_extents': [5, 0.05, 0.25]},   # Front wall
    {'position': [0, -5, 0.25], 'half_extents': [5, 0.05, 0.25]},  # Back wall
    {'position': [-5, 0, 0.25], 'half_extents': [0.05, 5, 0.25]},  # Left wall
    {'position': [5, 0, 0.25], 'half_extents': [0.05, 5, 0.25]},   # Right wall
    ]