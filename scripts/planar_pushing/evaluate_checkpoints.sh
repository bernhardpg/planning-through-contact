#!/bin/bash

# Set the path to the directory containing checkpoints
checkpoint_dir="/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v2/checkpoints/checkpoints_to_test"

# Set the path to your Python script
python_script="scripts/planar_pushing/run_sim_actuated_cylinder_diffusion.py"

# Get a list of checkpoints in reverse alphabetical order
checkpoints=($(ls -r "${checkpoint_dir}"/*.ckpt))

# Iterate through checkpoints
for checkpoint in "${checkpoints[@]}"; do
    # Extract the filename without the path and extension
    checkpoint_filename=$(basename -- "$checkpoint")
    checkpoint_name="${checkpoint_filename%.*}"

    # Run the Python script with the --checkpoint argument
    python "${python_script}" --checkpoint "${checkpoint}" --num_runs 20 --max_attempt_duration 60 --seed 9001

    echo "Evaluated checkpoint: ${checkpoint_name}"
done