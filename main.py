import argparse

from examples.large_scale import gcs_a_star
from examples.one_d_pusher import plan_for_box_pickup, plan_for_box_pushing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", help="Demo to run", type=str, default="box_pickup")
    args = parser.parse_args()
    demo_to_run = args.demo

    if demo_to_run in "box_push":
        plan_for_box_pushing()
    elif demo_to_run in "box_pickup":
        plan_for_box_pickup()
    elif demo_to_run in "a_star":
        gcs_a_star()
    return 0


if __name__ == "__main__":
    main()
