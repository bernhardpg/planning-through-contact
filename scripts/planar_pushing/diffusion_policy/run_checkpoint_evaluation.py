import argparse
import os
import subprocess


def main(checkpoint_dir, config_name):
    # Set the path to your Python script
    python_script = "scripts/planar_pushing/diffusion_policy/run_sim_diffusion.py"

    # Get a list of checkpoints in reverse alphabetical order
    checkpoints = sorted(os.listdir(checkpoint_dir), reverse=True)

    # Iterate through checkpoints
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        # Check if the item in the directory is a file and has a .ckpt extension
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".ckpt"):
            # Run the Python script with the --checkpoint argument
            command = f"python {python_script} --config-name {config_name} diffusion_policy_config.checkpoint=\\'{checkpoint_path}\\'"
            os.system(command)
            print("Evaluated checkpoint:", os.path.splitext(checkpoint)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints in reverse alphabetical order."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the directory containing checkpoints",
    )
    parser.add_argument(
        "--config-name", type=str, required=True, help="Name of the config file"
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    config_name = args.config_name

    if not os.path.exists(checkpoint_dir):
        print("Checkpoint directory does not exist:", checkpoint_dir)
    else:
        main(checkpoint_dir, config_name)
