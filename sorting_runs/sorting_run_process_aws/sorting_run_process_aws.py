import os
import numpy as np
import json
from copy import deepcopy

import psutil
import multiprocessing as mp
from tqdm import tqdm
import itertools

from sorting_run_one_aws import ex

dt = 0.01
n_steps = 5_000
# n_steps = 10
# n_frames=2
n_frames = 500

## todo: just rerun the runs that did not work..
with open('../../reruns/rerun_params_lst.json', 'r') as f:
    params_lst = json.load(f)

chunksize = 1
# Extract default tissue parameters
tp_dict = deepcopy(ex.configurations[0]._conf["tissue_params"])
ap_dict = deepcopy(ex.configurations[0]._conf["active_params"])
sp_dict = deepcopy(ex.configurations[0]._conf["simulation_params"])


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
    ap["v0"] = v0
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
    p.cpu_affinity([cpu_id])  # Set the CPU affinity to the specified CPU ID
    target(*args)


if __name__ == '__main__':
    max_processes = 15  # Maximum number of parallel processes
    active_processes = []
    cpu_count = psutil.cpu_count()

    for i, params in enumerate(params_lst):
        cpu_id = i % cpu_count  # Assign a CPU core based on process index

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
        process.join()


# if __name__ == '__main__':
#     max_processes = 5  # Maximum number of parallel processes
#     active_processes = []
#
#     for params in params_lst[:1]:
#         # Start new processes while the limit is not reached
#         if len(active_processes) < max_processes:
#             process = mp.Process(target=do_sim_in_parallel, args=(params,))
#             process.start()
#             active_processes.append(process)
#         else:
#             # Wait for at least one process to finish before starting a new one
#             while len(active_processes) >= max_processes:
#                 for p in active_processes:
#                     if not p.is_alive():
#                         active_processes.remove(p)
#
#             # Start a new process after making room
#             process = mp.Process(target=do_sim_in_parallel, args=(params,))
#             process.start()
#             active_processes.append(process)
#
#     # Wait for all processes to complete
#     for process in active_processes:
#         process.join()


# # Parallelize simulation
# if __name__ == '__main__':
#     processes = []
#
#     for params in params_lst:
#         process = mp.Process(target=do_sim_in_parallel, args=(params,))
#         processes.append(process)
#         process.start()
#
#     for process in processes:
#         process.join()
