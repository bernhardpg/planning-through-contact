import numpy as np
import zarr
import pickle
import pathlib
import argparse
import math
import os

from PIL import Image
from tqdm import tqdm

from planning_through_contact.visualize.analysis import (
    CombinedPlanarPushingLogs,
    PlanarPushingLog,
)

def main():
    """
    Converts data generated from run_sim_actuated_cylinder_camera.py to zarr format.
    
    data_dir: str - path to directory containing data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    data_dir = pathlib.Path(args.data_dir)
    debug = args.debug

    concatenated_states = []
    concatenated_actions = []
    concatenated_images = []
    concatenated_targets = []
    episode_ends = []
    current_end = 0
    for plan in tqdm(os.listdir(data_dir)):
        traj_dir = data_dir.joinpath(plan)
        if not os.path.isdir(traj_dir):
            continue
       
        image_dir = traj_dir.joinpath("images")
        log_path = traj_dir.joinpath("combined_planar_pushing_logs.pkl")
        
        # load pickle file and timing variables
        combined_logs = pickle.load(open(log_path, 'rb'))
        pusher_desired = combined_logs.pusher_desired
        # slider_desired = combined_logs.slider_desired

        freq = 10.0
        dt = 1 / freq
        t = combined_logs.pusher_desired.t
        total_time = math.floor(t[-1] * freq) / freq
        
        # get start time
        start_idx = get_start_idx(pusher_desired)   
        start_time = math.ceil(t[start_idx]*freq) / freq

        # get state, action, images
        state = []
        current_time = start_time
        idx = start_idx
        images = []
        while current_time < total_time:
            # state and action
            idx = get_closest_index(t, current_time, idx)
            current_state = np.array([pusher_desired.x[idx], 
                                    pusher_desired.y[idx], 
                                    pusher_desired.theta[idx]
            ])
            state.append(current_state)
        
            # image
            # This line can be simplified but it is clearer this way.
            # Image names are "{time in ms}" rounded to the nearest 100th
            image_name = round((current_time * 1000) / 100) * 100
            image_path = image_dir.joinpath(f"{int(image_name)}.png")
            img = Image.open(image_path).convert('RGB')
            img = np.asarray(img)
            images.append(img)
            if debug:
                from matplotlib import pyplot as plt
                print(f"\nCurrent time: {current_time}")
                print(f"Current index: {idx}")
                print(f"Image path: {image_path}")
                print(f"Current state: {current_state}")
                plt.imshow(img[6:-6, 6:-6, :])
                plt.show()

            # update current time
            current_time = round((current_time + dt) * freq) / freq

        state = np.array(state) # T x 3
        action = np.array(state)[:,:2] # T x 2
        action = np.concatenate([action[1:, :], action[-1:, :]], axis=0)
        images = np.array(images)

        # get target
        target = np.array([0.5, 0.0, 0.0]) # TODO: this is hardcoded
        target = np.array([target for _ in range(len(state))])

        # update concatenated arrays
        concatenated_states.append(state)
        concatenated_actions.append(action)
        concatenated_images.append(images)
        concatenated_targets.append(target)
        episode_ends.append(current_end + len(state))
        current_end += len(state)

    # save to zarr
    zarr_path = data_dir.joinpath("planning_through_contact.zarr")
    root = zarr.open_group(zarr_path, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # Chunk sizes optimized for read (not for supercloud storage, sorry admins)
    state_chunk_size = (1024, 3)
    action_chunk_size = (2048, 2)
    target_chunk_size = (1024, 3)
    image_chunk_size = (128, *images[0].shape)

    # convert to numpy
    concatenated_states = np.concatenate(concatenated_states, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    concatenated_images = np.concatenate(concatenated_images, axis=0)
    concatenated_targets = np.concatenate(concatenated_targets, axis=0)
    episode_ends = np.array(episode_ends)
    assert episode_ends[-1] == concatenated_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_actions.shape[0]
    assert concatenated_states.shape[0] == concatenated_images.shape[0]
    assert concatenated_states.shape[0] == concatenated_targets.shape[0]

    data_group.create_dataset('state', 
                              data=concatenated_states, 
                              chunks=state_chunk_size
    )
    data_group.create_dataset('action', 
                              data=concatenated_actions, 
                              chunks=action_chunk_size
    )
    data_group.create_dataset('img', 
                              data=concatenated_images, 
                              chunks=image_chunk_size
    )
    data_group.create_dataset('target', 
                              data=concatenated_targets, 
                              chunks=target_chunk_size
    )
    meta_group.create_dataset('episode_ends', 
                              data=episode_ends
    )

def get_start_idx(pusher_desired):
    """
    Finds the index of the first "non-stationary" command
    """
    length = len(pusher_desired.t)
    first_non_zero_idx = 0
    for i in range(length):
        if pusher_desired.x[i] != 0 or pusher_desired.y[i] != 0 or pusher_desired.theta[i] != 0:
            first_non_zero_idx = i
            break
    
    initial_state = np.array([
        pusher_desired.x[first_non_zero_idx], 
        pusher_desired.y[first_non_zero_idx], 
        pusher_desired.theta[first_non_zero_idx]
    ])
    assert not np.allclose(initial_state, np.array([0.0, 0.0, 0.0]))

    for i in range(first_non_zero_idx+1, length):
        state = np.array([pusher_desired.x[i], pusher_desired.y[i], pusher_desired.theta[i]])
        if not np.allclose(state, initial_state):
            return i
    
    return None
    
    

def get_closest_index(arr, t, start_idx=None, end_idx=None):
    """
    Returns index of arr that is closest to t.
    """
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(arr)
    
    min_diff = float('inf')
    min_idx = -1
    eps = 1e-4
    for i in range(start_idx, end_idx):
        diff = abs(arr[i] - t)
        if diff > min_diff:
            return min_idx
        if diff < eps:
            return i
        if diff < min_diff:
            min_diff = diff
            min_idx = i

if __name__ == "__main__":
    main()