from examples.one_d_pusher import (
    plan_for_one_box_one_finger,
    plan_for_one_finger_two_boxes,
    plan_for_two_fingers,
)


def main():
    plan = "two_fingers"
    if plan == "one_finger_one_box":
        plan_for_one_box_one_finger()
    elif plan == "two_fingers":
        plan_for_two_fingers()
    elif plan == "two_boxes":
        plan_for_one_finger_two_boxes()
    return 0


if __name__ == "__main__":
    main()
