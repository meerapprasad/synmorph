import numpy as np
from numba import jit

## todo: make v0 a function of cell type ...
##  todo: instead of changing v0, should I change dt ..?

class ActiveForce:
    """
    Active force class
    ------------------

    Calculates the active forces acting on a cell centroid. 
    This is traditionally phrased in terms of v0 and Dr, being the fixed velocity 
    and the rotational diffusion of the direction.
    """

    def __init__(self, tissue, active_params=None):
        assert active_params is not None, "Specify active params"
        self.t = tissue
        self.active_params = active_params
        self.aF = None
        self.orientation = np.random.uniform(0, np.pi * 2, self.t.mesh.n_c)
        # self.k = 0
        self.nts = self.active_params["tfin"] / self.active_params["dt"]
        if "anneal_v0_params" in self.active_params:
            self.active_params["v0"] = anneal_v0(self.active_params["anneal_v0_params"], self.nts)
        elif type(self.active_params["v0"]) is float:
            self.active_params["v0"] = self.active_params["v0"] * np.ones(self.t.mesh.n_c)
        self.get_active_force(k=0)
        # todo: take this from simulation instead of reinitializing
        if "angle0" not in self.active_params:
            self.active_params["angle0"] = 0
        if "alpha_dir" not in self.active_params:
            self.active_params["alpha_dir"] = 0

    # def update_active_param(self, param_name, val):
    #     self.active_params[param_name] = val

    def update_orientation(self, dt):
        """
        Time-steps the orientation (angle of velocity) according to the equation outlined in Bi et al PRX.
        :param dt:
        :return:
        """
        # self.orientation = _update_persistent_random_orientation(self.orientation,
        #                                                    self.active_params["Dr"],
        #                                                    dt,
        #                                                    self.t.mesh.n_c)
        self.orientation = _update_persistent_directional_orientation(self.orientation,
                                                           self.active_params["Dr"],
                                                           dt,
                                                           self.t.mesh.n_c,
                                                           self.active_params["alpha_dir"],
                                                           self.active_params["angle0"],)


    @property
    def orientation_vector(self):
        """
        Property. Converts angle to a unit vector
        :return: Unit vector
        """
        return _vec_from_angle(self.orientation)

    def get_active_force(self, k):
        """
        Standard SPV model
        :return:
        """
        # todo: pass in the step number here
        # todo: indexing is off when I do runga kutta
        # print(k)
        self.aF = _get_active_force(self.orientation_vector,
                                    self.active_params["v0"][k])

        ## todo: k affects dt ...
    def update_active_force(self, dt, k):
        self.update_orientation(dt)
        self.get_active_force(k)
        return self.aF

    ##but could include other options here...


@jit(nopython=True)
def _get_active_force(orientation, v0):
    return (v0 * orientation.T).T


@jit(nopython=True)
def _update_persistent_random_orientation(orientation, Dr, dt, n_c):
    return (orientation + np.random.normal(0, np.sqrt(2 * Dr * dt), n_c))%(np.pi*2)

@jit(nopython=True)
def _update_persistent_directional_orientation(orientation, Dr, dt, n_c,alpha_dir,angle0):
    return (orientation + np.random.normal(0, np.sqrt(2 * Dr * dt), n_c))%(np.pi*2) \
           - dt*alpha_dir*((orientation-angle0 + np.pi)%(np.pi*2) - np.pi)


@jit(nopython=True)
def _vec_from_angle(vec):
    return np.column_stack((np.cos(vec), np.sin(vec)))


def anneal_v0(anneal_v0_params, steps):
    """ anneal movement from init_v0 to final_v0 over steps """
    init_v0, perc_linear_time, final_v0 = (anneal_v0_params['init_v0'],
                                           anneal_v0_params['perc_linear_time'],
                                           anneal_v0_params['final_v0'])
    ## prioritize linear end temp over percent linear time
    if 'linear_end_temp' in anneal_v0_params:
        linear_end_temp = anneal_v0_params['linear_end_temp']
    else:
        linear_end_temp = init_v0 * (1 - perc_linear_time)

    # Calculate the decay rate based on the end_temp and the length of the exponential decay
    exp_decay_rate = -np.log(final_v0 / linear_end_temp) / (steps - int(perc_linear_time * steps))

    linear_decay = np.linspace(init_v0, linear_end_temp, int(perc_linear_time * steps))
    exp_decay = linear_end_temp * np.exp(-exp_decay_rate * np.arange(0, steps - len(linear_decay)))
    return np.concatenate((linear_decay, exp_decay))