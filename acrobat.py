import numpy as np
from tiling import tile
from interface import Domain

ACTIONS = [-1, 0, 1]


class Acrobat(Domain):

    def __init__(
            self,
            tile_width,
            L1=1,
            L2=1,
            m1=1,
            m2=1,
            LC1=0.5,
            LC2=0.5,
            g=9.8,
            tau=0.05,
            goal_height=None,
            xp1=0,
            yp1=0
    ):
        self.tile_width = tile_width
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.LC1 = LC1
        self.LC2 = LC2
        self.g = g
        self.tau = tau
        self.state = None
        self.goal_height = goal_height if goal_height is not None else self.LC1
        self.xp1 = xp1
        self.yp1 = yp1
        self.coordinates = None

    def get_init_state(self):
        state = np.array([0, 0, 0, 0])
        self.state = state
        self.coordinates = self.compute_coordinates()

        return tile(state=self.state, w=self.tile_width), ACTIONS

    def get_current_state(self):
        return tile(state=self.state, w=self.tile_width)

    def get_child_state(self, action):

        # intermediate computations
        theta1, theta1_dot, theta2, theta2_dot = self.state
        phi2 = self.m2 * self.LC2 * self.g * np.cos(theta1 + theta2 - (np.pi / 2))
        phi1 = -self.m2 * self.L1 * self.LC2 * theta2_dot ** 2 * np.sin(
            theta2) - 2 * self.m2 * self.L1 * self.LC2 * theta2_dot * theta1_dot * np.sin(
            theta2) + (self.m1 * self.LC1 + self.m2 * self.L1) * self.g * np.cos(theta1 - (np.pi / 2)) + phi2
        d2 = self.m2 * (self.LC2 ** 2 + self.L1 * self.LC2 * np.cos(theta2)) + 1
        d1 = self.m1 * self.LC1 ** 2 + self.m2 * (
                self.L1 ** 2 + self.LC2 ** 2 + 2 * self.L1 * self.LC2 * np.cos(theta2)) + 2
        theta2_dot_dot = (self.m2 * self.LC2 ** 2 + 1 - (d2 ** 2 / d1)) ** -1 * (
                    action + (d2 / d1) * phi1 - self.m2 * self.L1 * self.LC2 * theta1_dot ** 2 * np.sin(theta2) - phi2)
        theta1_dot_dot = -(d2 * theta2_dot_dot + phi1) / d1

        # update state variables
        theta2_dot += self.tau * theta2_dot_dot
        theta1_dot += self.tau * theta1_dot_dot
        theta2 += self.tau * theta2_dot
        theta1 += self.tau * theta1_dot

        # update state
        self.state = np.array([theta1, theta1_dot, theta2, theta2_dot])

        # update coordinates
        self.coordinates = self.compute_coordinates()

        return tile(state=self.state, w=self.tile_width), ACTIONS

    def is_current_state_terminal(self):
        y_tip = self.coordinates[-1]
        return y_tip >= self.goal_height

    def visualise(self):
        # TODO
        pass

    def compute_coordinates(self):
        theta1, theta1_dot, theta2, theta2_dot = self.state
        theta3 = theta1 + theta2
        xp2 = self.xp1 + self.L1 * np.sin(theta1)
        yp2 = self.yp1 - self.L1 * np.cos(theta1)
        x_tip = xp2 + self.L2 * np.sin(theta3)
        y_tip = yp2 - self.L2 * np.cos(theta3)
        return [self.xp1, self.yp1, xp2, yp2, x_tip, y_tip]