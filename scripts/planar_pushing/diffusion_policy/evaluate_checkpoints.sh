#!/bin/bash

# Initialize variables
checkpoint_dir=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--checkpoint-dir) checkpoint_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if checkpoint_dir is set
if [ -z "$checkpoint_dir" ]; then
    echo "Checkpoint directory is not provided. Use -d or --checkpoint-dir to specify the checkpoint directory."
    exit 1
fi

# Set the path to your Python script
python_script="scripts/planar_pushing/diffusion_policy/run_sim_diffusion.py"

# Get a list of checkpoints in reverse alphabetical order
checkpoints=($(ls -r "${checkpoint_dir}"/*.ckpt))

# Iterate through checkpoints
for ((i=${#checkpoints[@]}-1; i>=0; i--)); do
    checkpoint="${checkpoints[i]}"
    # Extract the filename without the path and extension
    checkpoint_filename=$(basename -- "$checkpoint")
    checkpoint_name="${checkpoint_filename%.*}"

    # Run the Python script with the --checkpoint argument
    python "${python_script}" --config-name "${config_name}" diffusion_policy_config.checkpoint="${checkpoint}"

    echo "Evaluated checkpoint: ${checkpoint_name}"
done
