from uuid import uuid4
import os
import h5py
import numpy as np
from glob import glob
from synmorph.analysis.connected_components import cc_per_dt
# from synmorph.analysis.topological import count_connected_components
from synmorph.simulation import Simulation
from synmorph.simulation import Simulation

#### This script runs one simulation and optionally stores
####   the experiment information (outputs and metadata) in
####   an experiment (`ex`) object that is managed by the
####   provenance system Sacred. Sacred will then store the
####   run metadata/info for subsequent retrieval. The
####   default parameters for the run are specified in a
####   configuration file in the "*run_one.py" script.


# Use a unique directory name for this run
uid = str(uuid4())

# Write to temporary directory of choice (fast read/write)
data_dir = os.path.abspath(f"/tmp/{uid}")  # Use root temp dir (Linux/MacOS)
# data_dir = f"/home/pbhamidi/scratch/lateral_signaling/tmp/{uid}"  # Scratch dir on Caltech HPC
os.makedirs(data_dir, exist_ok=True)


##### TODO: there may be a memory problem when I run other integration schemes
def do_one_simulation(ex=None, save_data=False, animate=False, **cfg):

    # Create simulation with defined configuration
    try:
        sim = Simulation(**cfg)
        # Run
        sim.simulate(progress_bar=True)
    ## todo: add an exception for timeout process error
    except Exception as e1:
        try:
            modified_cfg = cfg.copy()
            modified_cfg['simulation_params']['int_method'] = 'rk2'
            sim = Simulation(**modified_cfg)
            # Run
            sim.simulate(progress_bar=True)
            # sacred_storage_dir = os.path.abspath("./sacred")
        except Exception as e2:
            modified_cfg = cfg.copy()
            modified_cfg['simulation_params']['int_method'] = 'rk4'
            sim = Simulation(**modified_cfg)
            sim.simulate(progress_bar=True)

    if ex is not None:

        # Save any source code dependencies to Sacred
        source_files = glob(os.path.join("../synmorph", "*.py"))
        source_files = [os.path.abspath(f) for f in source_files]
        for sf in source_files:
            ex.add_source_file(sf)

        # Initialize stuff to save
        artifacts = []

        # Dump data to file
        if save_data:

            print("Writing data")

            data_dump_fname = os.path.join(data_dir, "results.npz")

            # Save connected components
            cc_arr = cc_per_dt(sim.t.c_types, sim.tri_save)
            # todo: compute perturb cell dist travelled, and elasticity of the tissue

            # Save data in compressed NPZ format
            np.savez_compressed(data_dump_fname,
                                c_types=sim.t.c_types,
                                t_span_save=sim.t_span_save,
                                tri_save=sim.tri_save,
                                x_save=sim.x_save,
                                cc_arr=cc_arr)

            # Add to Sacred artifacts
            artifacts.append(data_dump_fname)

        # Make animation
        if animate:

            print("Making animation")

            anim_fname = os.path.join(data_dir, "animation.mp4")
            sim.animate_c_types(
                dir_name=data_dir,
                file_name=anim_fname,
                c_type_col_map=cfg["c_type_col_map"],
                n_frames=cfg["n_frames"],
                fps=cfg["fps"],
                dpi=cfg["dpi"],
            )

            # Add to Sacred artifacts
            artifacts.append(anim_fname)

        # Add all artifacts to Sacred
        for _a in artifacts:
            ex.add_artifact(_a)
    else:
        return sim

# Dump data to an HDF5 file
# data_dump_fname = os.path.join(data_dir, "results.hdf5")
# # save connected components
# cc_arr = cc_per_dt(sim.t.c_types, sim.tri_save)
#
# with h5py.File(data_dump_fname, "w") as f:
#     # f.create_dataset("c_types", data=sim.t.c_types, compression="gzip")
#     # f.create_dataset("t_span_save", data=sim.t_span_save, compression="gzip")
#     # f.create_dataset("tri_save", data=sim.tri_save, compression="gzip")
#     # f.create_dataset("x_save", data=sim.x_save, compression="gzip")
#     # f.create_dataset("cc_arr", data=cc_arr, compression="gzip")
#     f.create_dataset("c_types", data=sim.t.c_types)
#     f.create_dataset("t_span_save", data=sim.t_span_save)
#     f.create_dataset("tri_save", data=sim.tri_save)
#     f.create_dataset("x_save", data=sim.x_save)
#     f.create_dataset("cc_arr", data=cc_arr)


        # modified_cfg["simulation_params"]["dt"] = cfg["simulation_params"]["dt"] /10
        # # modified_cfg["simulation_params"]["tfin"] = cfg["simulation_params"]["tfin"] * 2
        # if 'anneal_v0_params' in cfg['active_params']:
        #     modified_cfg['active_params']['dt'] = cfg['active_params']['dt'] / 10
        #     # modified_cfg['active_params']['tfin'] = cfg['active_params']['tfin'] * 2