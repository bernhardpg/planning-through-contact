import argparse

from experiments.one_d_pusher import plan_for_box_pickup, plan_for_box_pushing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", help="Demo to run", type=str, default="box_pickup")
    args = parser.parse_args()
    demo_to_run = args.demo

    if demo_to_run in "box_push":
        plan_for_box_pushing()
    elif demo_to_run in "box_pickup":
        plan_for_box_pickup()
    else:
        raise NotImplementedError(f"Demo of type {demo_to_run} not implemented.")

    return 0


if __name__ == "__main__":
    main()
