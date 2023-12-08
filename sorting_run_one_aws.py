import os
import sacred
from glob import glob
from sacred.observers import FileStorageObserver
from sorting_simulation_logic import do_one_simulation
from sacred import SETTINGS
# SETTINGS['CAPTURE_MODE'] = 'sys'
import datetime as dt

# Set up Sacred experiment
ex = sacred.Experiment("sorting-test-aws")

# Save any source code dependencies to Sacred
source_files = glob(os.path.join("synmorph", "*.py"))
source_files = [os.path.abspath(f) for f in source_files]
for sf in source_files:
    ex.add_source_file(sf)

# Set storage location for all Sacred results
res_dir = "./sacred"                          # Local

# Set default experimental configuration
config_file = os.path.abspath("configs/no-perturb-aws.json")

@ex.config_hook
def custom_config_hook(config, command_name, logger):
    """Custom config hook to set up a custom directory for each run based on specified config keys."""
    # Generate a custom directory name based on keys_to_log
    proj_dir = config["expt_name"]
    # todo: consider adding date here, but not good if I run something overnight.
    custom_dir = f"{res_dir}/{proj_dir}/"

    # Remove previous FileStorageObservers to avoid duplicate logging
    ex.observers = [obs for obs in ex.observers if not isinstance(obs, FileStorageObserver)]

    # Add a FileStorageObserver with the custom path for this run
    ex.observers.append(FileStorageObserver(custom_dir))
    return config


ex.add_config(config_file)

## todo: add a seed here bc I will need to avg over several params
@ex.main  # Use ex as our provenance system and call this function as __main__()
def run_one_simulation(_config, _run, seed):
    """Simulates SPV given a single parameter configuration"""
    # _config contains all the variables in the configuration
    # _run contains data about the run
    c = _config.copy()

    # Extract save_data and remove from config
    # save_data = c.pop('save_data')
    # animate_data = c.pop('animate_data')
    do_one_simulation(
        ex=ex,
        **_config
    )

    ## todo: try-except block here that only changes dt
