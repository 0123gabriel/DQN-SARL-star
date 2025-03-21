import numpy as np

class Env:
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 31
        self.motions = np.array([(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)])*0.1
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range

        # Road data
        small_radius = 7
        big_radius = 13.6

        obs = set()

        for i in np.linspace(0, big_radius):
            obs.add((i, -small_radius))
        for i in np.linspace(0, big_radius):
            obs.add((i, -big_radius))
        for i in np.linspace(-big_radius, -small_radius):
            obs.add((big_radius, i))

        lower_circle_angles = [np.pi/2, 3*np.pi/2]
        lower_center = (0, 0)
        angles_lower_circle = np.linspace(lower_circle_angles[0], lower_circle_angles[1], 100)

        for i in range(len(angles_lower_circle) - 1):
            x1 = small_radius*np.cos(angles_lower_circle[i]) + lower_center[0]
            y1 = small_radius*np.sin(angles_lower_circle[i]) + lower_center[1]
            x2 = small_radius*np.cos(angles_lower_circle[i+1]) + lower_center[0]
            y2 = small_radius*np.sin(angles_lower_circle[i+1]) + lower_center[1]
            obs.add((x1, y1))
            obs.add((x2, y2))

            x3 = big_radius*np.cos(angles_lower_circle[i]) + lower_center[0]
            y3 = big_radius*np.sin(angles_lower_circle[i]) + lower_center[1]
            x4 = big_radius*np.cos(angles_lower_circle[i+1]) + lower_center[0]
            y4 = big_radius*np.sin(angles_lower_circle[i+1]) + lower_center[1]
            obs.add((x3, y3))
            obs.add((x4, y4))

        upper_circle_angles = (-np.pi/2, np.pi/2)
        upper_center = (0, 20.6)
        angles_upper_circle = np.linspace(upper_circle_angles[0], upper_circle_angles[1], 100)

        for i in range(len(angles_upper_circle) - 1):
            x5 = small_radius*np.cos(angles_upper_circle[i]) + upper_center[0]
            y5 = small_radius*np.sin(angles_upper_circle[i]) + upper_center[1]
            x6 = small_radius*np.cos(angles_upper_circle[i+1]) + upper_center[0]
            y6 = small_radius*np.sin(angles_upper_circle[i+1]) + upper_center[1]
            obs.add((x5, y5))
            obs.add((x6, y6))

            x7 = big_radius*np.cos(angles_upper_circle[i]) + upper_center[0]
            y7 = big_radius*np.sin(angles_upper_circle[i]) + upper_center[1]
            x8 = big_radius*np.cos(angles_upper_circle[i+1]) + upper_center[0]
            y8 = big_radius*np.sin(angles_upper_circle[i+1]) + upper_center[1]
            obs.add((x7, y7))
            obs.add((x8, y8))


        for i in np.linspace(-big_radius, 0):
            obs.add((i, big_radius + 2*small_radius))
        for i in np.linspace(-big_radius, 0):
            obs.add((i, 2*big_radius + small_radius))
        for i in np.linspace(big_radius + 2*small_radius, 2*big_radius + small_radius):
            obs.add((-big_radius, i))

        return obs
