import _pickle as cPickle
import bz2
import pickle
import json

import os
import numpy as np
from numba import jit

import synmorph.tri_functions as trf
from synmorph.active_force import ActiveForce
from synmorph.force import Force
from synmorph.mesh import Mesh
from synmorph import utils

# todo: add perturbation params here

class Tissue:
    """
    Tissue class
    ------------

    This class sits above the mesh, force, active_force,grn classes and integrates information about geometry and other properties of cells to determine forces.

    This class is used to initialize the mesh, force and active_force classes.

    """

    def __init__(self, tissue_params=None, active_params=None, init_params=None, initialize=True, calc_force=True, meshfile=None,
                 run_options=None,tissue_file=None):

        if tissue_file is None:
            assert tissue_params is not None, "Specify tissue params"
            assert active_params is not None, "Specify active params"
            assert init_params is not None, "Specify init params"
            assert run_options is not None, "Specify run options"
            self.tissue_params = tissue_params
            self.init_params = init_params

            # todo: initialize W from a config
            # Convert adhesion matrix to numpy array
            if 'init_config_file' not in init_params:
            # if "W" in self.tissue_params.keys():
                self.tissue_params.update({"W": np.asarray(self.tissue_params["W"])})
            elif 'perturb_W' in self.tissue_params:
                with open(os.path.join(init_params["init_config_file"] + "config.json"), 'r') as file:
                    config_ = json.load(file)
                rest_W = np.asarray(config_['tissue_params']["W"])
                ## todo: make a new W with perturb_W
                perturb_W = np.asarray(self.tissue_params['perturb_W'])
                W = np.zeros((perturb_W.shape[1],perturb_W.shape[1]))
                W[:2, :2] = rest_W
                W[2:] = perturb_W
                # W[0, 2], W[1,2] = perturb_W[0][0], perturb_W[0][1]
                self.tissue_params.update({"W": W})

            self.mesh = None

            self.c_types = None
            self.nc_types = None
            self.c_typeN = None
            self.tc_types, self.tc_typesp, self.tc_typesm = None, None, None

            if meshfile is None:
                assert init_params is not None, "Must provide initialization parameters unless a previous mesh is parsed"
                if initialize:
                    self.initialize(run_options)
            else:
                self.mesh = Mesh(load=meshfile, run_options=run_options)
                assert self.L == self.mesh.L, "The L provided in the params dict and the mesh file are not the same"

            for par in ["A0", "P0", "kappa_A", "kappa_P"]:
                self.tissue_params[par] = _vectorify(self.tissue_params[par], self.mesh.n_c)

            self.active = ActiveForce(self, active_params)

            if calc_force:
                self.get_forces()
            else:
                self.F = None

            self.time = None

            self.name = None
            self.id = None

        else:
            self.load(tissue_file)

    def set_time(self, time):
        """
        Set the time and date at which the simulation was performed. For logging.
        :param time:
        :return:
        """
        self.time = time

    def update_tissue_param(self, param_name, val):
        """
        Short-cut for updating a tissue parameter
        :param param_name: dictionary key
        :param val: corresponding value
        :return:
        """
        self.tissue_params[param_name] = val

    def initialize(self, run_options=None):
        """
        Initialize the tissue. Here, the mesh is initialized, and cell types are assigned.
        In the future, this may want to be generalized.

        :param run_options:
        :return:
        """
        if self.init_params["init_config_file"] is not None:
            # c_arr = np.load(self.init_params["init_config_file"])
            results_ = np.load(self.init_params["init_config_file"] + "results.npz")
            self.initialize_mesh(x=results_['x_save'][-1], run_options=run_options)
            self.assign_ctypes_from_config(results_['c_types'],  n_perturb=self.init_params["n_perturb"])

        else:
            self.initialize_mesh(x=None, run_options=run_options)
            self.assign_ctypes(self.init_params["sorted"])


    def initialize_mesh(self, x=None, run_options=None):
        """
        Make initial condition. Currently, this is a hexagonal lattice + noise

        Makes reference to the self.hexagonal_lattice function, then crops down to the reference frame

        If x is supplied, this is over-ridden

        :param run_options:
        :param L: Domain size/length (np.float32)
        :param noise: Gaussian noise added to {x,y} coordinates (np.float32)
        """
        if x is None:
            x = trf.hexagonal_lattice(int(self.L), int(np.ceil(self.L)), noise=self.init_noise, A=self.A0)
            x += 1e-3
            np.argsort(x.max(axis=1))
            x = x[np.argsort(x.max(axis=1))[:int(self.L ** 2 / self.A0)]]

        self.mesh = Mesh(x, self.L, run_options=run_options)

    # todo: seed here
    def assign_ctypes(self, sorted=False):  #, mixed=False
        if sorted:
            self.assign_ctypes_sorted()
        else:
            assert sum(self.c_type_proportions) == 1.0, "c_type_proportions must sum to 1.0"
            assert (np.array(self.c_type_proportions) >= 0).all(), "c_type_proportions values must all be >=0"
            self.nc_types = len(self.c_type_proportions)
            self.c_typeN = [int(pr * self.mesh.n_c) for pr in self.c_type_proportions[:-1]]
            self.c_typeN += [self.mesh.n_c - sum(self.c_typeN)]
            c_types = np.zeros(self.mesh.n_c, dtype=np.int32)
            j = 0
            for k, ctN in enumerate(self.c_typeN):
                c_types[j:j + ctN] = k
                j += ctN
            np.random.shuffle(c_types)
            self.c_types = c_types
            self.c_type_tri_form()

    def assign_ctypes_sorted(self):
        # Calculate the midpoint of the x-coordinates
        ## todo: change to update this based on input
        midpoint_x = np.mean(self.mesh.x[:, 0])

        # Assign labels based on the midpoint
        labels = [0 if x < midpoint_x else 1 for x, _ in self.mesh.x]
        self.c_types = np.array(labels).astype(int)
        self.nc_types = len(np.unique(self.c_types))
        self.c_typeN = np.unique(self.c_types, return_counts=True)[1].tolist()
        self.c_type_tri_form()

    def assign_ctypes_from_config(self, c_type_arr, n_perturb=None):  #todo: add perturbation here?
        self.c_types = c_type_arr.astype(np.int32)
        if n_perturb is not None:
            # todo: center of cell type 1, perturb outwards n_perturb
            c2_points = self.mesh.x[c_type_arr == 1]
            centroid = np.mean(c2_points, axis=0)
            distances = np.linalg.norm(c2_points - centroid, axis=1)
            closest_index = np.argmin(distances)
            closest_point = c2_points[closest_index]
            perturb_idx = np.argwhere(self.mesh.x == closest_point)[0][0]
            self.perturb_idx = perturb_idx
            self.c_types[perturb_idx] = 2

        self.nc_types = len(np.unique(self.c_types))
        self.c_typeN = np.unique(self.c_types, return_counts=True)[1].tolist()
        self.c_type_tri_form()

    def c_type_tri_form(self):
        """
        Convert the nc x 1 c_type array to a nv x 3 array -- triangulated form.
        Here, the CW and CCW (p,m) cell types can be easily deduced by the roll function.
        :return:
        """
        self.tc_types = trf.tri_call(self.c_types, self.mesh.tri)
        self.tc_typesp = trf.roll(self.tc_types, -1)
        self.tc_typesm = trf.roll(self.tc_types, 1)

    def get_forces(self):
        """
        Calculate the forces by calling the Force class.
        :return:
        """
        self.F = Force(self).F
        return sum_forces(self.F, self.active.aF)

    # todo: specify step as an index here
    def update(self, dt, k):
        """
        Wrapper for update functions.
        :param dt: time-step.
        :return:
        """
        self.update_active(dt, k)
        self.update_mechanics()

    def update_active(self, dt, k):
        """
        Wrapper for update of active forces
        :param dt: time-step
        :return:
        """
        self.active.update_active_force(dt, k)

    def update_mechanics(self):
        """
        Wrapper of update of the mesh. The mesh is retriangulated and the geometric properties are recalculated.

        Then the triangulated form of the cell types are reassigned.
        :return:
        """
        self.mesh.update()
        self.c_type_tri_form()

    def update_x_mechanics(self, x):
        """
        Like update_mechanics, apart from x is explicitly provided.
        :param x:
        :return:
        """
        self.mesh.x = x
        self.update_mechanics()

    @property
    def init_noise(self):
        return float(self.init_params["init_noise"])

    @property
    def c_type_proportions(self):
        return self.init_params["c_type_proportions"]

    @property
    def L(self):
        return self.tissue_params["L"]

    @property
    def A0(self):
        return self.tissue_params["A0"]

    @property
    def P0(self):
        return self.tissue_params["P0"]

    @property
    def kappa_A(self):
        return self.tissue_params["kappa_A"]

    @property
    def kappa_P(self):
        return self.tissue_params["kappa_P"]

    @property
    def W(self):
        return self.tissue_params["W"]

    @property
    def a(self):
        return self.tissue_params["a"]

    @property
    def k(self):
        return self.tissue_params["k"]

    ###More properties, for plotting primarily.

    @property
    def dA(self):
        return self.mesh.A - self.A0

    @property
    def dP(self):
        return self.mesh.P - self.P0

    def get_latex(self, val):
        if val in utils._latex:
            return utils._latex[val]
        else:
            print("No latex conversion in the dictionary.")
            return val

    def save(self, name, id=None, dir_path="", compressed=False):
        
        dir_path = os.path.abspath(dir_path)  # Returns current directory if empty string
        dir_path = os.makedirs(dir_path, exist_ok=True)  # Makes dir if it doesn't exist
        fname    = os.path.join(dir_path, self.name + "_tissue")

        self.name = name
        
        if id is None:
            self.id = {}
        else:
            self.id = id
        
        if compressed:
            with bz2.BZ2File(fname + '.pbz2', 'w') as f:
                cPickle.dump(self.__dict__, f)
        else:
            with open(fname + '.pickle', 'wb') as pikd:
                pickle.dump(self.__dict__, pikd)

    def load(self, fname):
        if fname.split(".")[1] == "pbz2":
            fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))

        else:
            pikd = open(fname, 'rb')
            fdict = pickle.load(pikd)
            pikd.close()
        self.__dict__ = fdict


@jit(nopython=True)
def sum_forces(F, aF):
    return F + aF


@jit(nopython=True)
def _vectorify(x, n):
    return x * np.ones(n)

    # When there are two or three cell types, sort them without shuffling.
#     j = 0
#     for k, ctN in enumerate(self.c_typeN[:2]):
#         c_types[j:j + ctN] = k
#         j += ctN
#     if self.nc_types == 3:
#         # When there are three cell types, insert the third one in the center of the second cell type section.
#         second_type_start = self.c_typeN[0]
#         second_type_end = second_type_start + self.c_typeN[1]
#         middle_index = (second_type_start + second_type_end) // 2
#         # Insert the third cell type in the middle of the second cell type section.
#         c_types[middle_index] = 2
#         # Adjust counts for the second and third cell types.
#         c_types[second_type_end:] = 1
#
# self.c_types = c_types
# self.c_type_tri_form()
# Splitting the frame vertically into two halves
# Assuming cells are stored row-wise in the array
# for row in range(self.L):
#     for col in range(self.L):
#         # Calculate the linear index from 2D coordinates
#         index = row * self.L + col
#         # Assign cell type based on the column
#
#         if col < self.L // 2:
#             c_types[index] = 0  # First cell type for the left half
#         else:
#             c_types[index] = 1  # Second cell type for the right half
# self.c_types = c_types.reshape((self.L, self.L))  # Reshape if needed for further processing
# self.c_type_tri_form()


   # c_types = np.zeros(self.mesh.n_c, dtype=np.int32)
   #              new_cell_type = 2  # Assuming 0 and 1 are the other two types
   #              num_new_cells = self.c_typeN[2]  # Number of cells for the new cell type
   #              # Calculate the side length of the square for the new cell type
   #              side_length = int(np.ceil(np.sqrt(num_new_cells)))
   #              # Calculate center coordinates for the left half (cell type 0)
   #              center_row = self.L // 2
   #              center_col = self.L // 4  # Middle of the left half
   #              # Determine starting point for placing the new cell type
   #              start_row = int(max(center_row - side_length // 2, 0))
   #              start_col = int(max(center_col - side_length // 2, 0))
   #              width, height = self.L, self.L
   #              # Assign cell types
   #              new_cell_count = 0
   #              for row in range(start_row, min(start_row + side_length, height)):
   #                  for col in range(start_col, min(start_col + side_length, width // 2)):
   #                      if new_cell_count < num_new_cells:
   #                          index = row * width + col
   #                          c_types[index] = new_cell_type
   #                          new_cell_count += 1
   #              # Fill in remaining cells with the existing cell types
   #              for row in range(height):
   #                  for col in range(width):
   #                      index = row * width + col
   #                      if c_types[index] == 0:
   #                          if col < width // 2:
   #                              c_types[index] = 0  # First cell type for the left half
   #                          else:
   #                              c_types[index] = 1


# else:
# if self.nc_types == 3:
#     lattice = self.create_hexagonal_lattice()
#     c_types = np.ones(len(lattice), dtype=int)
#     mid_x = self.L * 3 / 4
#     c_types[lattice[:, 0] < mid_x] = 0
#
#     # Find center of the left side for the special cell type
#     center_x = mid_x / 2
#     center_y = self.L * np.sqrt(3) / 2
#     distances = np.sqrt((lattice[:, 0] - center_x) ** 2 + (lattice[:, 1] - center_y) ** 2)
#     closest_indices = np.argsort(distances)[:self.c_typeN[2]]
#     c_types[closest_indices] = 2
# self.c_types = c_types  # .reshape((height, width))  # Reshape for further processing
# self.c_type_tri_form()

    # def create_hexagonal_lattice(self):
    #     width, height = self.L, self.L
    #     lattice = []
    #     for row in range(height):
    #         for col in range(width):
    #             x = col * 3/2  # 3/2 comes from hexagonal lattice geometry
    #             y = np.sqrt(3) * (row + 0.5 * (col % 2))  # Offset every other column
    #             lattice.append((x, y))
    #     return np.array(lattice)