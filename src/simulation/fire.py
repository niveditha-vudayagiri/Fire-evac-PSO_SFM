import numpy as np

class Fire:
    def __init__(self, origin):
        self.origin = origin
        self.radius = 0

    def grow(self, step):
        if self.radius > 15:
            return
        self.radius = 0.1 * step

    def get_fire_points(self):
        angles = np.linspace(0, 2 * np.pi, 50)
        x = self.origin['x'] + self.radius * np.cos(angles)
        y = self.origin['y'] + self.radius * np.sin(angles)
        z = [self.origin['z']] * len(x)
        return x, y, z
