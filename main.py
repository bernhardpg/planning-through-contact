import argparse

from examples.one_d_pusher import (
    plan_for_one_box_one_finger,
    plan_for_one_finger_two_boxes,
    plan_for_two_fingers,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", help="Demo to run", type=str, default="two_fingers")
    args = parser.parse_args()
    demo_to_run = args.demo

    if demo_to_run in "one_finger_one_box":
        plan_for_one_box_one_finger()
    elif demo_to_run in "two_fingers":
        plan_for_two_fingers()
    elif demo_to_run in "two_boxes":
        plan_for_one_finger_two_boxes()
    return 0


if __name__ == "__main__":
    main()
