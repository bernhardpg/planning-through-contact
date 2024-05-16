import os
from datetime import datetime

from tqdm import tqdm


def run_data_generation_script(config_name, plans_dir):
    seed = datetime.now().timestamp()
    command = (
        f"python scripts/planar_pushing/diffusion_policy/run_data_generation.py "
        f"--config-name {config_name} "
        f"data_collection_config.plans_dir={plans_dir} "
        f"data_collection_config.plan_config.seed={int(seed) % 1000} "
        # f"multi_run_config.seed={int(seed) % 1000} "
        f"data_collection_config.rendered_plans_dir=trajectories_rendered/three_boxes_tmp"
    )

    os.system(command)


# Specify the range of indices
start_index = 0
end_index = 3  # Adjust this value as needed

# Specify the config name
config_name = "real_iiwa_sim_config.yaml"

if __name__ == "__main__":
    # Loop through the indices and execute the command
    for i in tqdm(
        range(start_index, end_index + 1),
        total=end_index - start_index + 1,
        desc="Running data generation script",
    ):
        plans_dir = f"trajectories/three_boxes/run_{i}"
        print(plans_dir)
        run_data_generation_script(config_name, plans_dir)
