import os
import shutil

# Path to the directory containing subdirectories
base_dir = "trajectories/single_box200"

# Number of subdirectories to move into each "run" directory
batch_size = 100

# Create the "run" directories if they don't exist
for i in range(0, len(os.listdir(base_dir)), batch_size):
    run_dir = os.path.join(base_dir, f"run_{i // batch_size}")
    os.makedirs(run_dir, exist_ok=True)

# Move subdirectories into the "run" directories
i = 0
for subdir in sorted(
    [name for name in os.listdir(base_dir) if not name.endswith("yaml")],
    key=lambda x: int(x.split('_')[1])
):
    if subdir.startswith("traj"):
        source = os.path.join(base_dir, subdir)
        destination = os.path.join(base_dir, f"run_{i // batch_size}")
        shutil.move(source, destination)
        i += 1

print("All subdirectories moved into their respective 'run' directories.")
