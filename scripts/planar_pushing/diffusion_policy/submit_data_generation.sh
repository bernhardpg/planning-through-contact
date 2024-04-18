#!/bin/bash

# Parse command line arguments
# Usage: ./submit_data_generation.sh --config-name CONFIG_NAME [NUM_NODE, NUM_PROC, NUM_THREAD]
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--config-name) CONFIG_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if CONFIG_NAME is set
if [ -z "$CONFIG_NAME" ]; then
    echo "Configuration name is not provided. Use -c or --config-name to specify the configuration name."
    exit 1
fi

# Initialize and Load Modules
cd ../../../. # Go to the root directory of the project
source /etc/profile
poetry shell
export PYTHONPATH=~/workspace/drake-build-bernhard/install/lib/python3.10/site-packages:~/workspace/gcs-diffusion:

echo $LLSUB_RANK
echo $LLSUB_SIZE

# Run Python script with provided configuration name
python scripts/planar_pushing/diffusion_policy/run_data_generation.py --config-name "$CONFIG_NAME" \
    data_collection_config.LLSUB_RANK=$LLSUB_RANK \
    data_collection_config.LLSUB_SIZE=$LLSUB_SIZE