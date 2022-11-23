import math
import matplotlib.pyplot as plt
import numpy as np


def tile(state: np.ndarray, bounds: np.ndarray, bins: int, num_of_tilings=None, displacements=None, visualize=False):
    """
    Each tiling splits a domain comprising continuous values into multiple tiles. The number of
    tiles is determined by the *bins* parameter. Each tiling is offset by a specific amount on each
    axis. The offset depends on the width of a bin, the number of tilings, as well as the displacement
    factor.

    :param state: state comprising *n* dimensions (ndarray<num_of_dimensions>)
    :param displacements: displacement of a tiling in each dimension (ndarray<num_of_dimensions>)
    :param num_of_tilings: number of tilings
    :param bins: number of bins for each dimension
    :param bounds: lower and upper boundary for each dimension
    :param visualize: Visualizes the grid if true
    :return: encoded state based on the tilings
    """
    if num_of_tilings is None:
        # default to the **minimum** number of tilings
        num_of_tilings = math.ceil(2 ** np.log2(4 * len(bounds)))
    if displacements is None:
        # default to displacements recommended by the book: the first odd numbers
        displacements = np.array([2 * k - 1 for k in range(1, len(bounds) + 1)])

    tile_widths = [(upper - lower) / bins for lower, upper in bounds]
    offsets = [width / num_of_tilings for width in tile_widths]

    tiling_shape = np.full((len(bounds)), bins)
    tilings = np.zeros((num_of_tilings, *tiling_shape))

    # create bins per dimension
    bins_per_dim = np.array([
        # NOTE: don't include the upper bound, as we want points exactly on this boundary to be INCLUDED in the last bin
        np.linspace(boundary[0], boundary[1], bins + 1)[1:-1]
        for boundary, width in zip(bounds, tile_widths)
    ])

    # mark overlapping region in each tile
    for i in range(num_of_tilings):
        # displace state negatively, which is the same as displacing the tile positively
        displaced_state = state - displacements * offsets * i
        # find indexes in  tile (one index per dimension of state space)
        indexes = [
            np.digitize(dim_value, bins)
            for dim_value, bins in zip(displaced_state, bins_per_dim)
        ]
        # recall that indexing with a tuple works like this: a[(1,2,3)] == a[1][2][3]
        tilings[i][tuple(indexes)] = 1

    if visualize:
        visualize_grid(
            num_of_tilings=num_of_tilings,
            bounds=bounds,
            displacements=displacements,
            tile_widths=tile_widths,
            offsets=offsets
        )
        plt.plot(state[0], state[1], 'o')
        plt.show()

    return tilings


def visualize_grid(
        num_of_tilings: int,
        bounds: list[list[float]],
        displacements: np.ndarray,
        tile_widths: np.ndarray | list,
        offsets: np.ndarray | list
):

    # create bins per dimension
    multi_grid = []
    for i in range(num_of_tilings):
        grid = []
        for boundary, width, displacement, offset in zip(bounds, tile_widths, displacements, offsets):
            lower, upper = boundary + displacement * offset * i
            grid.append(
                list(np.arange(lower+width, upper, width))
            )
        multi_grid.append(grid)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    line_styles = ['-', '--', ':']
    lines = []
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(multi_grid):
        for x in grid[0]:
            l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], label=i)
        for y in grid[1]:
            l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)])
        lines.append(l)
    ax.grid(False)
    ax.legend(lines, ["Tiling #{}".format(t) for t in range(len(lines))], facecolor='white')
    ax.set_title("Tilings")
    return ax


if __name__ == '__main__':
    sample_state = np.array([-1, 1])
    t = tile(sample_state, np.array([[-1, 1], [-1, 1]]), 10, 1, np.array([1, 1]), visualize=True)
    print(t.shape)
    print(t)
