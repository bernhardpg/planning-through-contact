import pickle

import numpy as np


def main():
    output_file = "arbitrary_shape_pickles/small_t_pusher.pkl"
    boxes = [
        {
            "name": "box",
            "size": [0.2, 0.05, 0.05],
            "transform": np.eye(4),
        },
        {
            "name": "box",
            "size": [0.05, 0.15001, 0.05],  # Require a small overlap
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -0.1],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    ]

    with open(output_file, "wb") as f:
        pickle.dump(boxes, f)


if __name__ == "__main__":
    main()
