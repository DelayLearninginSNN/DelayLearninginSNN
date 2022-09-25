import numpy as np
from matplotlib import pyplot as plt

PRE_WINDOW = -10
POST_WINDOW = 7
W=16
REPETITIVE_LENGTH = 20000
CONVERGENT_LENGTH = 20000
DIVERGENT_LENGTH = 2000
SATURATION_LENGTH = 10000
STD_THRESHOLD = 0.1
CORRELATION_THRESHOLD = 0.95
SLOPE_THRESHOLD = 0.001
MAX_DELAY = 40
MIN_DELAY = 0.1
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
temp_c = COLORS[1]
COLORS[1] = COLORS[2]
COLORS[2] = temp_c
rng = np.random.default_rng(1)
COLORS += [(rng.random(), rng.random(), rng.random()) for x in range(1000)]
DCAT_COLORS = ["green", "red", "blue","yellow"]
DELAY_CATEGORIES = ["converging", "repeating", "uncategorized", "increasing", "decreasing", "min", "max"]
DELAY_CATEGORIES_SHORTLIST = ["converging", "diverging", "repeating", "uncategorized", "dormant"]
CATEGORY_CONVERSION = {
    "converging": "converging",
    "repeating": "repeating",
    "uncategorized": "uncategorized",
    "max": "diverging",
    "min": "diverging",
    "decreasing": "diverging",
    "increasing": "diverging"
}

