import dataclasses

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tiling import tile
from interface import Domain
from typing import Union

ACTIONS = [-1, 0, 1]


@dataclasses.dataclass
class AnimationFrame:
    """
    Everything needed to render a single frame of animation
    """
    xp1: float
    yp1: float
    xp2: float
    yp2: float
    x_tip: float
    y_tip: float
    # one can not choose an action for terminal state, so allow None
    action: Union[float, None]
    step: int


def frame_properties(f: AnimationFrame):
    return f.xp1, f.yp1, f.xp2, f.yp2, f.x_tip, f.y_tip, f.action, f.step


class Acrobat(Domain):

    def __init__(
            self,
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
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.LC1 = LC1
        self.LC2 = LC2
        self.g = g
        self.tau = tau
        self.state = None
        self.goal_height = goal_height if goal_height is not None else self.L1
        self.xp1 = xp1
        self.yp1 = yp1
        self.frames = []

        # tiling configuration
        # using bounds recommended by sutton-1996
        theta1_dot_bound = 4 * np.pi
        theta_2_dot_bound = 9 * np.pi
        self.bounds = np.array([
            [-self.tau * theta1_dot_bound, self.tau * theta1_dot_bound],
            [-theta1_dot_bound, theta1_dot_bound],
            [-self.tau * theta_2_dot_bound, self.tau * theta_2_dot_bound],
            [-theta_2_dot_bound, theta_2_dot_bound]
        ])
        # using number of bins recommended by sutton-1996
        self.bins = 6
        # using number of tilings recommended by sutto-1996
        self.tilings = 48

    def get_init_state(self):
        state = np.array([0, 0, 0, 0])
        self.state = state
        self.frames = [
            AnimationFrame(
                *self.compute_coordinates(),
                action=None,
                step=0,
            )
        ]

        return tile(self.state, bounds=self.bounds, num_of_tilings=self.tilings, bins=self.bins).flatten(), ACTIONS

    def get_current_state(self):
        return tile(self.state, bounds=self.bounds, num_of_tilings=self.tilings, bins=self.bins).flatten()

    def get_child_state(self, action):
        # update latest frame with chosen action
        self.frames[-1].action = action

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

        # add frame
        self.frames.append(
            AnimationFrame(
                *self.compute_coordinates(),
                action=None,
                step=len(self.frames),
            )
        )

        return tile(self.state, bounds=self.bounds, num_of_tilings=self.tilings, bins=self.bins).flatten(), ACTIONS

    def is_current_state_terminal(self):
        y_tip = self.frames[-1].y_tip
        return y_tip >= self.goal_height

    def visualise(self, filename=None):
        # source: https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/

        # setup animation
        fig, ax = plt.subplots(figsize=(5, 5))
        xlim = (-3, 3)
        ylim = (-3, 3)
        ax.set(xlim=xlim, ylim=ylim)

        ax.plot(xlim, [self.goal_height, self.goal_height], linestyle='dotted')
        xp1, yp1, xp2, yp2, x_tip, y_tip, action, step = frame_properties(self.frames[0])
        acrobat_line = ax.plot([xp1, xp2, x_tip], [yp1, yp2, y_tip], color='k', linewidth=2)[0]
        acrobat_text = ax.text(
            xlim[0] + 0.1,
            ylim[1] - 0.1,
            f"action: {action}\nstep: {step}",
            horizontalalignment='left',
            verticalalignment='top',
        )

        def animate(i):
            xp1, yp1, xp2, yp2, x_tip, y_tip, action, step = frame_properties(self.frames[i])
            acrobat_line.set_xdata([xp1, xp2, x_tip])
            acrobat_line.set_ydata([yp1, yp2, y_tip])
            acrobat_text.set_text(f"action: {action}\nstep: {step}")

        anim = FuncAnimation(
            fig=fig,
            func=animate,
            frames=range(1, len(self.frames)),
            interval=25,
        )

        if filename is not None:
            anim.save(filename)

        plt.draw()
        plt.show()

    def compute_coordinates(self):
        theta1, theta1_dot, theta2, theta2_dot = self.state
        theta3 = theta1 + theta2
        xp2 = self.xp1 + self.L1 * np.sin(theta1)
        yp2 = self.yp1 - self.L1 * np.cos(theta1)
        x_tip = xp2 + self.L2 * np.sin(theta3)
        y_tip = yp2 - self.L2 * np.cos(theta3)
        return [self.xp1, self.yp1, xp2, yp2, x_tip, y_tip]


if __name__ == '__main__':
    # demonstration of the acrobat problem: implement a simple "pumping" agent that just tries to maximize the current
    # angular velocity of the top-most joint at every step (solves the problem about 300 steps)
    ac = Acrobat()
    ac.get_init_state()
    while not ac.is_current_state_terminal():
        theta1_dot = ac.state[1]
        if theta1_dot < 0:
            action = 1
        else:
            action = -1
        ac.get_child_state(action)
    ac.visualise(filename="demo.gif")
