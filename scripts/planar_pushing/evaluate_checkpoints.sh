#!/bin/bash

# Set the path to the directory containing checkpoints
checkpoint_dir="/home/adam/workspace/gcs-diffusion/data/outputs/push_box_v2/checkpoints"

# Set the path to your Python script
python_script="scripts/planar_pushing/run_sim_actuated_cylinder_diffusion.py"

# Get a list of checkpoints in reverse alphabetical order
checkpoints=($(ls -r "${checkpoint_dir}"/*.ckpt))

# Iterate through checkpoints
for ((i=${#checkpoints[@]}-1; i>=0; i--)); do
    checkpoint="${checkpoints[i]}"
    # Extract the filename without the path and extension
    checkpoint_filename=$(basename -- "$checkpoint")
    checkpoint_name="${checkpoint_filename%.*}"

    # Run the Python script with the --checkpoint argument
    python "${python_script}" --checkpoint "${checkpoint}" --num-runs 10 --max-attempt-duration 90 --seed 9001

    echo "Evaluated checkpoint: ${checkpoint_name}"
done

# # Iterate through --traj_idx values from 5 to 50
# for traj_idx in {6..99}; do
#     # Run the Python script with the --traj_idx argument
#     # uses default checkpoint, seed, etc
#     python "${python_script}" --num_runs 1 --traj_idx "${traj_idx}"

#     # Optionally, you can print a message for reference
#     echo "Evaluated --traj_idx: ${traj_idx}"