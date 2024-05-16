import pickle
import numpy as np


def main():
    output_file = "data/arbitrary_shape.pickle"
    boxes = [
        {
            "name": "box",
            "size": [0.3, 0.1, 0.05],
            "transform": np.eye(4),
        },
        {
            "name": "box",
            "size": [0.15, 0.15001, 0.05],
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, 0.15],
                    [0.0, 1.0, 0.0, -0.125],
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
