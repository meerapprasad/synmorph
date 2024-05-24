from copy import deepcopy
import math
import json
from run_one_sim import ex

dt = 0.01  #args.dt
n_steps = 750  # args.n_steps
n_frames = int(n_steps/4)     # args.n_frames

# Extract default tissue parameters
tp_dict = deepcopy(ex.configurations[0]._conf["tissue_params"])
ap_dict = deepcopy(ex.configurations[0]._conf["active_params"])
sp_dict = deepcopy(ex.configurations[0]._conf["simulation_params"])

## todo: if loading from a previous run, override these params
w_ab = math.exp(-1.44)
# v0 = np.linspace(1.1, 2.1, 5)
v0 = 5.0
p0 = 3.7
global_seed = 100
_params = [w_ab, v0, p0]
# todo: update the seed here
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
        seed=global_seed
    )
    ex.run(config_updates=cfg_updates)

do_sim_in_parallel(_params)



print('done')