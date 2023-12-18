from copy import deepcopy
import argparse
from sorting_runs.sorting_run_parallel.sorting_run_one import ex

# todo: implement computing connected components at each step and saving
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_ab', type=float, default=0.0072, help='cross-adhesion strength')
    parser.add_argument('--v0', type=float, default=0.1, help='cell velocity')
    parser.add_argument('--p0', type=float, default=3.81, help='target cell shape index')
    parser.add_argument('--dt', type=float, default=0.025, help='integration step size')
    parser.add_argument('--n_steps', type=int, default=100, help='number of time steps')
    parser.add_argument('--n_frames', type=int, default=10, help='number of frames to plot')
    return parser.parse_args()


# Extract default tissue parameters
tp_dict = deepcopy(ex.configurations[0]._conf["tissue_params"])
ap_dict = deepcopy(ex.configurations[0]._conf["active_params"])
sp_dict = deepcopy(ex.configurations[0]._conf["simulation_params"])


def do_sim_in_parallel(args):
    """Perform a sim on one worker."""
    # Make adhesion matrix
    _w = [[0., args.w_ab], [args.w_ab, 0.]]

    # Update tissue params dict
    tp = deepcopy(tp_dict)
    tp["W"] = _w
#    tp["kappa_P"] = kappa_p
    tp["P0"] = args.p0
    ap = deepcopy(ap_dict)
    ap["v0"] = args.v0
    sp = deepcopy(sp_dict)
    sp["dt"] = args.dt
    sp["tfin"] = args.n_steps

    # Run with modified configurations
    cfg_updates = dict(
        tissue_params=tp,
        active_params=ap,
        simulation_params=sp,
        n_frames=args.n_frames,
    )
    ex.run(config_updates=cfg_updates)

## todo: do multiprocessing here because I/O errors.
# run one simulation at a time, this does not parallelize
if __name__ == '__main__':
    args = get_args()
    do_sim_in_parallel(args)
