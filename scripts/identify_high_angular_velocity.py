import os
import pickle

import numpy as np


# Function to compute angular speed
def compute_angular_speed(time, orientation):
    dt = np.diff(time)
    dtheta = np.diff(orientation)
    angular_speed = abs(dtheta / dt)

    # Remove sharp angular velocity at beginning
    first_zero_idx = -1
    for i in range(len(angular_speed)):
        if np.allclose(angular_speed[i], 0.0):
            first_zero_idx = i
            break

    return angular_speed[first_zero_idx:]


# Function to identify high angular speed moments
def identify_high_angular_speed(angular_speed, threshold, window_size):
    angular_speed_cumsum = np.cumsum(angular_speed)
    for i in range(len(angular_speed_cumsum) - window_size):
        window_avg = (
            angular_speed_cumsum[i + window_size] - angular_speed_cumsum[i]
        ) / window_size
        if window_avg > threshold:
            return True
    return False


if __name__ == "__main__":
    # Root directory containing subdirectories with demonstration data
    root_dir = "trajectories_rendered/test_angular_speed"
    # Threshold for high angular speed (this can be adjusted)
    angular_speed_threshold = 2.0
    # Number of consecutive time steps for high angular speed
    window_size = 5

    # List to keep track of subdirectories with high angular velocities
    subdirs_with_high_angular_speed = []

    # Loop through subdirectories in the root directory
    # subdirs = []
    # for subdir in os.listdir(root_dir):
    #     if os.path.isdir(os.path.join(root_dir, subdir)):
    #         subdirs.append(subdir)
    # subdirs = sorted(subdir, key=lambda x: int(x))
    subdirs = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path) and "config.yaml" not in subdir:
            subdirs.append(subdir)
    subdirs.sort(key=lambda x: int(x))

    for subdir in subdirs:
        # Path to the pickle file
        subdir_path = os.path.join(root_dir, subdir)
        traj_log_path = os.path.join(subdir_path, "combined_logs.pkl")

        # Load the pickle file
        with open(traj_log_path, "rb") as f:
            combined_logs = pickle.load(f)

        # Extract time and orientation data
        slider_desired = combined_logs.slider_desired
        time = slider_desired.t
        orientation = slider_desired.theta

        # Compute angular speed
        angular_speed = compute_angular_speed(time, orientation)
        print(max(angular_speed))

        # Identify high angular speed moments
        has_high_angular_speed = identify_high_angular_speed(
            angular_speed, angular_speed_threshold, window_size
        )

        # If high angular speed moments are found, add the subdirectory to the list
        if has_high_angular_speed:
            subdirs_with_high_angular_speed.append(int(subdir))
            print(f"High angular speed moments found in {subdir}")

    # Print the names of all subdirectories that contained high angular velocities
    print("\nSubdirectories with high angular velocities:")
    subdirs_with_high_angular_speed.sort()
    for subdir in subdirs_with_high_angular_speed:
        print(subdir)
