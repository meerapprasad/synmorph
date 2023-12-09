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
n_frames = 500

## todo: just rerun the runs that did not work..
with open('reruns/rerun_params_lst.json', 'r') as f:
    params_lst = json.load(f)

# ## param vals to scan
# w_range = [-1, -3]
# # kappa_p_range = [0.05, 0.15]
# v0_range = [0.0, 0.15]
# p0_range = [3.5, 3.9]
#
# ## todo: read params from file
# n_vals = 12
# w_vals = [float(w) for w in np.logspace(w_range[0], w_range[1], n_vals)]
# # kappa_p_vals = [float(k) for k in np.linspace(kappa_p_range[0], kappa_p_range[1], n_vals)]
# v0_vals = [float(v) for v in np.linspace(v0_range[0], v0_range[1], n_vals)]
# p0_vals = [float(p) for p in np.linspace(p0_range[0], p0_range[1], n_vals)]

# Set chunksize for workers

# chunksize = 10
# todo: set chunksize dynamically based on n_jobs/n_cpus
# chunksize = int(np.ceil(n_vals ** 3/ os.cpu_count()))
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


# Parallelize simulation
if __name__ == '__main__':
    # Get param values to change
    # params_lst =   #list(itertools.product(w_vals, v0_vals, p0_vals))[160:] #kappa_p_vals,
    print("Assembling worker pool")

    # Get worker pool
    # pool = mp.Pool(psutil.cpu_count(logical=False))
    pool = mp.Pool(10)
    print("Performing parallel simulations")

    # Run in parallel
    results = pool.imap(do_sim_in_parallel, params_lst, chunksize=chunksize)

    pool.close()
    pool.join()
