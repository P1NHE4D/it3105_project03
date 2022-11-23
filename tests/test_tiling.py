import unittest

import numpy as np

from tiling import tile

displacements = np.array([1, 1])
boundaries = np.array([[-1, 1], [-1, 1]])
bins = 3
tilings = 2


class MyTestCase(unittest.TestCase):

    def test_corners(self):
        state = np.array([-10, -10])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]

        ]
        self.assertEqual(gt, t)

        state = np.array([10, 10])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]
            ]

        ]
        self.assertEqual(gt, t)

        state = np.array([-10, 10])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0]
            ]

        ]
        self.assertEqual(gt, t)

        state = np.array([10, -10])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0]
            ]

        ]
        self.assertEqual(gt, t)

    def test_center(self):
        state = np.array([0, 0])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ]

        ]
        self.assertEqual(gt, t)

    def test_asymmetric_dims(self):
        state = np.array([-10, 0])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]

        ]
        self.assertEqual(gt, t)

        state = np.array([0, -10])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0]
            ]

        ]
        self.assertEqual(gt, t)

        state = np.array([0, 10])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0]
            ]

        ]
        self.assertEqual(gt, t)

        state = np.array([10, 0])
        t = tile(state=state, bounds=boundaries, displacements=displacements, bins=bins, num_of_tilings=tilings)
        t = t.tolist()
        gt = [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0]
            ]

        ]
        self.assertEqual(gt, t)


if __name__ == '__main__':
    unittest.main()
