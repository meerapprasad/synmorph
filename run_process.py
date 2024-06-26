import os
import json
import numpy as np
import itertools
import math
from copy import deepcopy
import psutil
import multiprocessing as mp
from run_one_sim import ex, args

## todo: will need to update how v0 is saved
dt = 0.01  #args.dt
n_steps = 750    # args.n_steps
n_frames = int(n_steps/4)     # args.n_frames

## todo: specify a file with perturbation runs
if args.params_file is not None:
    with open(args.params_file, 'r') as f:
        # todo: specify which params here at some point
        #  make the params_file a numpy array
        params_lst = json.load(f)

# Extract default tissue parameters
tp_dict = deepcopy(ex.configurations[0]._conf["tissue_params"])
ap_dict = deepcopy(ex.configurations[0]._conf["active_params"])
sp_dict = deepcopy(ex.configurations[0]._conf["simulation_params"])

## todo: if loading from a previous run, override these params
w_ab = math.exp(-1.44)
# v0 = np.linspace(1.1, 2.1, 5)
v0 = 2.0
p0 = 3.10
# _params = list(itertools.product([w_ab], v0, [p0]))
_params = [[w_ab, v0, p0],]


def do_sim_in_parallel(params):
    """Perform a sim on one worker."""
    _wv, v0, p0 = params  #kappa_p,
    # Make adhesion matrix
    _w = [[0., _wv], [_wv, 0.]]

    # Update tissue params dict
    tp = deepcopy(tp_dict)
    tp["W"] = _w
#    tp["kappa_P"] = kappa_p
    tp["P0"] = p0
    ap = deepcopy(ap_dict)
    if 'v0' in ap:
        ap["v0"] = v0
    elif 'anneal_v0_params' in ap:
        ap['anneal_v0_params']['init_v0'] = v0
    elif 'perturb_v0_params' in ap:
        ap['perturb_v0_params']['p_v0'] = v0

    ap['tfin'] = n_steps
    ap['dt'] = dt
    sp = deepcopy(sp_dict)
    sp["dt"] = dt
    sp["tfin"] = n_steps

    # Run with modified configurations
    cfg_updates = dict(
        tissue_params=tp,
        active_params=ap,
        simulation_params=sp,
        n_frames=n_frames,
    )
    ex.run(config_updates=cfg_updates)


def start_process_with_affinity(target, args, cpu_id):
    p = psutil.Process(os.getpid())
    # p.cpu_affinity([cpu_id])  # Set the CPU affinity to the specified CPU ID
    p.cpu_affinity(cpu_id) ##if a list
    target(*args)

def calculate_cpu_ids(i, num_cpus_per_process, total_cpu_count):
    """
    Calculate a list of CPU IDs for a process based on its index and the desired number of CPUs per process.

    :param i: Index of the process.
    :param num_cpus_per_process: Number of CPUs to assign to each process.
    :param total_cpu_count: Total number of available CPUs.
    :return: List of CPU IDs.
    """
    cpu_ids = [(i * num_cpus_per_process + j) % total_cpu_count for j in range(num_cpus_per_process)]
    return cpu_ids


if __name__ == '__main__':
    max_processes = 3  # Maximum number of parallel processes
    active_processes = []
    cpu_count = psutil.cpu_count()

    for i, params in enumerate(_params):
        # cpu_id = i % cpu_count  # Assign a CPU core based on process index
        cpu_id = calculate_cpu_ids(i, 2, cpu_count)

        # Start new processes while the limit is not reached
        if len(active_processes) < max_processes:
            process = mp.Process(target=start_process_with_affinity, args=(do_sim_in_parallel, (params,), cpu_id))
            process.start()
            active_processes.append(process)
        else:
            # Wait for at least one process to finish before starting a new one
            while len(active_processes) >= max_processes:
                for p in active_processes:
                    if not p.is_alive():
                        active_processes.remove(p)

            process = mp.Process(target=start_process_with_affinity, args=(do_sim_in_parallel, (params,), cpu_id))
            process.start()
            active_processes.append(process)

    # Wait for all processes to complete
    for process in active_processes:
        process.join(timeout=90)
        if process.is_alive():
            print(f"Terminating process {process.pid} as it exceeded the timeout")
            process.terminate()
            print('done')