import math

import numpy as np


def tile(
        state: np.ndarray,
        w: float,
        bounds: np.ndarray,
        displacements:np.ndarray=None,
        n:int=None,
):
    """
    tile 'state' 'n' times', with the i'th tile consisting of regions of side-length 'w' bounded at 'bounds', and being
    displaced by w/n * displacements. As suggested by our RL intro book, we add an extra row and column of regions to
    the top and left of a tile, to ensure that a state always overlaps with every tile.

    :param state: vector describing the state as a real-numbered point in space
    :param w: length of side of len(state)-dimensional region of tile
    :param bounds: of shape (len(vector), 2) describes lower and upper bound for each dimension of state space, as a
                   unit of w
    :param displacements: vector with equal dimension to state describing the displacement of tiles
    :param n: number of tiles
    :return: tensor of shape (n, len(vector)) which coarse codes state
    """

    if n is None:
        # default to the **minimum** number of tiles for a given state dimensionality recommended by the book
        n = 2 ** np.log2(4 * state.shape[0])
    if displacements is None:
        # default to displacements recommended by the book: the first odd numbers
        displacements = np.array([2 * k - 1 for k in range(1, state.shape[0] + 1)])
    offset = w / n

    # find tile shape
    # for example with bounds=[(-3,3), (-2,1)] and w=2, the tile shape should be (3, 2)

    bounds_with_padding = [
        (lower-w, upper)
        for lower, upper in bounds
    ]

    state_spans = [
        upper - lower
        for lower, upper in bounds_with_padding
    ]

    tile_shape = [
        math.ceil(state_span / w)
        for state_span in state_spans
    ]

    tiles = np.zeros((n, *tile_shape))

    # create bins per dimension
    bins_per_dim = [
        # NOTE: don't include the upper bound, as we want points exactly on this boundary to be INCLUDED in the last bin
        list(np.arange(lower, upper, w))
        for lower, upper in bounds_with_padding
    ]

    # mark overlapping region in each tile
    for i in range(n):
        # displace state negatively, which is the same as displacing the tile positively
        state -= displacements * offset * i
        # find indexes in  tile (one index per dimension of state space)
        indexes = [
            np.digitize(dim_value, bins)-1
            for dim_value, bins in zip(state, bins_per_dim)
        ]
        # recall that indexing with a tuple works like this: a[(1,2,3)] == a[1][2][3]
        tiles[i][tuple(indexes)] = 1

    return tiles

if __name__ == '__main__':
    # basic sanity check against ground-truth calculated by hand
    state = np.array([0.0, 0.0])
    tiled = tile(
        state,
        w=2.0,
        bounds=[(-3.0,3.0), (-2.0,1.0)],
        displacements=np.array([1,1]),
        n=3,
    )

    assert np.array_equal(
        tiled,
        np.array(
            [
                [
                    [0.,0.,0.],
                    [0.,0.,0.],
                    [0.,0.,1.],  # 2,2    (tile is offset by 0,0)
                    [0.,0.,0.],
                ],
                [
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 1., 0.],  # 2,1    (tile is offset by 2/3,2/3)
                    [0., 0., 0.],
                ],
                [
                    [0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.],  # 1,1    (tile is offset by 4/3,4/3)
                    [0., 0., 0.],
                ],
            ]
        ),
    )