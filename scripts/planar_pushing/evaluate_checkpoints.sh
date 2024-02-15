#!/bin/bash

# Set the path to the directory containing checkpoints
checkpoint_dir="/home/adam/workspace/gcs-diffusion/data/outputs/push_tee_v2/checkpoints"

# Set the path to your Python script
python_script="scripts/planar_pushing/run_sim_actuated_cylinder_diffusion.py"

# Iterate through all checkpoints in the directory
for checkpoint in "${checkpoint_dir}"/*.ckpt; do
    # Extract the filename without the path and extension
    checkpoint_filename=$(basename -- "$checkpoint")
    checkpoint_name="${checkpoint_filename%.*}"

    # Run the Python script with the --checkpoint argument
    python "${python_script}" --checkpoint "${checkpoint}" --num_runs 20 --max_attempt_duration 50

    echo "Evaluated checkpoint: ${checkpoint_name}"
done