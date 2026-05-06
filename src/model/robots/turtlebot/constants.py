import re
import math
import numpy as np


START_POS = [0, 0, 0.05]
INIT_ORIENTATION = [0, 0, math.pi]

WHEEL_SEPARATION=0.16
WHEEL_RADIUS=0.033

MAX_FORCE=0.5

MAX_LIN_VEL = 0.22  # m/s
MAX_ANG_VEL = 2.84  # rad/s (160 degreea/s)