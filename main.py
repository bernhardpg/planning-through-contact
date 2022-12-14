import argparse

from examples.one_d_pusher import plan_for_box_pickup, plan_for_box_pushing
from examples.rotation import simple_rotations_test, sdp_relaxation , test_cuts, plot_cuts_corners_fixed, plot_cuts_with_fixed_position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", help="Demo to run", type=str, default="box_pickup")
    args = parser.parse_args()
    demo_to_run = args.demo

    if demo_to_run in "box_push":
        plan_for_box_pushing()
    elif demo_to_run in "box_pickup":
        plan_for_box_pickup()
    elif demo_to_run in "rotations":
        # simple_rotations_test()
        # sdp_relaxation()
        # test_cuts()
        plot_cuts_corners_fixed()
        # plot_cuts_with_fixed_position()
    return 0


if __name__ == "__main__":
    main()
