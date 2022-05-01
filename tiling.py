import math
import matplotlib.pyplot as plt
import numpy as np


class Tiling:

    def __init__(
            self,
            bounds: np.ndarray,
            bins: np.ndarray,
            displacements=None,
            tilings=None
    ):
        """
        Each tiling splits a domain comprising continuous values into multiple tiles. The number of
        tiles is determined by the *bins* parameter. Each tiling is offset by a specific amount on each
        axis. The offset depends on the width of a bin, the number of tilings, as well as the displacement
        factor.

        :param bounds: Boundaries of each dimension (ndarray<dimensions, lower, upper>)
        :param bins: number of bins, i.e., tiles, for each dimension (ndarray<dimensions>)
        :param displacements: displacement vector determining the spacing between two tilings (ndarray<tilings>)
        :param tilings: number of tilings, i.e., the number of grids
        """
        self.bounds = bounds
        self.bins = bins
        self.displacements = displacements
        self.tilings = tilings
        if self.tilings is None:
            # default to the **minimum** number of tilings
            self.tilings = math.ceil(2 ** np.log2(4 * bounds.shape[0]))
        if self.displacements is None:
            # default to displacements recommended by the book: the first odd numbers
            self.displacements = np.array([2 * k - 1 for k in range(1, bounds.shape[0] + 1)])
        self.multi_grid = self._construct_grid()

    def tile(self, state: np.ndarray, flatten=False, return_decoded=False):
        """
        :param state: state comprising *n* dimensions
        :param return_decoded: if true, decoded state is returned
        :param flatten: return flattened array
        :return: encoded state based on the tilings
        """
        encoded_state = []
        for grid in self.multi_grid:
            encoding = []
            for i, axis in enumerate(grid):
                if return_decoded:
                    encoding.append(np.digitize(state[i], axis))
                else:
                    encoded_axis = np.zeros(self.bins[i], dtype=int)
                    encoded_axis[np.digitize(state[i], axis)] = 1
                    encoding.extend(encoded_axis)
            encoded_state.append(encoding)
        encoded_state = np.array(encoded_state)
        if flatten:
            return encoded_state.flatten()
        return np.array(encoded_state)

    def _construct_grid(self):
        tile_widths = [(boundary[1] - boundary[0]) / (bins - 1) for boundary, bins in zip(self.bounds, self.bins)]
        offsets = [tile_width / self.tilings for tile_width in tile_widths]
        multi_grid = []
        for i in range(self.tilings):
            grid = []
            for j, boundary in enumerate(self.bounds):
                lower, upper = boundary
                grid.append(
                    np.linspace(lower, upper, self.bins[j] + 1)[1:-1] + i * self.displacements[j] * offsets[j]
                )
            multi_grid.append(grid)
        return multi_grid

    def visualize_grid(self):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        linestyles = ['-', '--', ':']
        legend_lines = []

        fig, ax = plt.subplots(figsize=(10, 10))
        for i, grid in enumerate(self.multi_grid):
            for x in grid[0]:
                l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
            for y in grid[1]:
                l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
            legend_lines.append(l)
        ax.grid(False)
        ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white',
                  framealpha=0.9)
        ax.set_title("Tilings")
        return ax


if __name__ == '__main__':
    # basic sanity check against ground-truth calculated by hand
    t = Tiling(np.array([[-1, 1], [-1, 1]]), np.array([10, 10]), tilings=2, displacements=[1, 1])
    ax: plt.Axes = t.visualize_grid()
    samples = np.array([
        [-0.5, -0.5],
        [0, 0],
        [0.5, 0.5],
        [1, 1],
        [-1, -1],
        [-0.75, 0.5]
    ])
    for sample in samples:
        for i, tiling in enumerate(t.tile(sample, return_decoded=True)):
            print("x={}, y={} | Tiling: {} | Tile: {}, {}".format(*sample, i, *tiling))
        ax.plot(sample[0], sample[1], marker='o', markersize=10)
    plt.show()
