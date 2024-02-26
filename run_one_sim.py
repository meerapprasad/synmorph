import os
import argparse
import sacred
from glob import glob
from sacred.observers import FileStorageObserver
from sorting_runs.sorting_simulation_logic import do_one_simulation
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

## todo: prevent cells from moving out of frame

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str, default='configs/anneal-temp.yml', help='config file')
    parser.add_argument('--sacred_expt', type=str, default='anneal-temp', help='experiment name')
    parser.add_argument('--dt', type=float, default=0.01, help='time step')
    parser.add_argument('--n_steps', type=int, default=100, help='number of time steps')
    parser.add_argument('--n_frames', type=int, default=10, help='number of frames to plot')
    parser.add_argument('--params_file', type=str, default=None, help='path to params to vary')
    return parser.parse_args()

args = get_args()
# Set up Sacred experiment
## todo: set the sacred expt name with kwargs
ex = sacred.Experiment(args.sacred_expt)

# Save any source code dependencies to Sacred
source_files = glob(os.path.join("../synmorph", "*.py"))
source_files = [os.path.abspath(f) for f in source_files]
for sf in source_files:
    ex.add_source_file(sf)

# Set storage location for all Sacred results
res_dir = "sacred"  # Local

# Set default experimental configuration
config_file = os.path.abspath(args.config_yaml)

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
    c = _config.copy()

    do_one_simulation(
        ex=ex,
        **_config
    )