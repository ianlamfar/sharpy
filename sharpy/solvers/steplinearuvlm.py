"""
Time domain solver to integrate the linear UVLM aerodynamic system developed by S. Maraniello
N Goizueta
Nov 18
"""
import os
import sys
from sharpy.utils.solver_interface import BaseSolver, solver
import numpy as np
import sharpy.utils.settings as settings
import sharpy.utils.generator_interface as gen_interface
import sharpy.linear.src.linuvlm as linuvlm  


@solver
class StepLinearUVLM(BaseSolver):
    """
    Warnings:
        Currently under development. The directory location has to be updated once the linear UVLM modules
        are included in SHARPy

    Time domain aerodynamic solver that uses a linear UVLM formulation

    To Do:
        Link documentation with Linear UVLM module. Add velocity generator settings

    Attributes:
    settings (dict): Contains the solver's ``settings``. See below for acceptable values:

        ====================  =========  ===============================================  ==========
        Name                  Type       Description                                      Default
        ====================  =========  ===============================================  ==========
        ``dt``                ``float``  Time increment                                   ``0.1``
        ``integr_order``      ``int``    Integration order for UVLM system (1 or 2).      ``2``
        ``ScalingDict``       ``dict``   Dictionary with scaling gains. See Notes.
        ``solution_method``   ``str``    UVLM linear system solution method. See Notes    ``direct``
        ``remove_predictor``  ``bool``   Remove predictor term from UVLM system assembly  ``True``
        ====================  =========  ===============================================  ==========


    """
    solver_id = 'StepLinearUVLM'

    def __init__(self):
        """
        Create default settings
        """

        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['dt'] = 'float'
        self.settings_default['dt'] = 0.1

        self.settings_types['integr_order'] = 'int'
        self.settings_default['integr_order'] = 2

        self.settings_types['density'] = 'float'
        self.settings_default['density'] = 1.225

        self.settings_types['ScalingDict'] = 'dict'
        self.settings_default['ScalingDict'] = {'length': 1.0,
                                                'speed': 1.0,
                                                'density': 1.0}

        self.settings_types['solution_method'] = 'str'
        self.settings_default['solution_method'] = 'direct'

        self.settings_types['remove_predictor'] = 'bool'
        self.settings_default['remove_predictor'] = True

        self.data = None
        self.settings = None
        self.lin_uvlm_system = None
        self.velocity_generator = None

    def initialise(self, data, custom_settings=None):
        r"""
        Initialises the Linear UVLM aerodynamic solver and the chosen velocity generator.

        Settings are parsed into the standard SHARPy settings format for solvers. It then checks whether there is
        any previous information about the linearised system (in order for a solution to be restarted without
        overwriting the linearisation).

        If a linearised system does not exist, a linear UVLM system is created linearising about the current time step.

        The reference values for the input and output are transformed into column vectors :math:`\mathbf{u}`
        and :math:`\mathbf{y}`, respectively.

        The information pertaining to the linear system is stored in a dictionary ``self.data.aero.linear`` within
        the main ``data`` variable.

        Args:
            data (PreSharpy): class containing the problem information
            custom_settings (dict): custom settings dictionary

        """

        self.data = data

        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        # Check whether linear UVLM has been initialised
        try:
            self.data.aero.linear
        except AttributeError:
            self.data.aero.linear = dict()
            aero_tstep = self.data.aero.timestep_info[-1]

            # TODO: verify of a better way to implement rho
            self.data.aero.timestep_info[-1].rho = self.settings['density'].value

            # Generate instance of linuvlm.Dynamic()
            lin_uvlm_system = linuvlm.Dynamic(aero_tstep,
                                              dt=self.settings['dt'].value,
                                              integr_order=self.settings['integr_order'].value,
                                              ScalingDict=self.settings['ScalingDict'],
                                              RemovePredictor=self.settings['remove_predictor'])

            # Save reference values
            # System Inputs
            zeta = np.concatenate([aero_tstep.zeta[i_surf].reshape(-1, order='C')
                                   for i_surf in range(aero_tstep.n_surf)])
            zeta_dot = np.concatenate([aero_tstep.zeta_dot[i_surf].reshape(-1, order='C')
                                       for i_surf in range(aero_tstep.n_surf)])
            u_ext = np.concatenate([aero_tstep.u_ext[i_surf].reshape(-1, order='C')
                                    for i_surf in range(aero_tstep.n_surf)])
            u_0 = np.concatenate((zeta, zeta_dot, u_ext))

            # Linearised state
            gamma = np.concatenate([aero_tstep.gamma[i_surf].reshape(-1, order='C')
                                    for i_surf in range(aero_tstep.n_surf)])
            gamma_star = np.concatenate([aero_tstep.gamma_star[i_surf].reshape(-1, order='C')
                                         for i_surf in range(aero_tstep.n_surf)])
            gamma_dot = np.concatenate([aero_tstep.gamma_dot[i_surf].reshape(-1, order='C')
                                        for i_surf in range(aero_tstep.n_surf)])
            x_0 = np.concatenate((gamma, gamma_star, gamma_dot))

            # Reference forces
            f_0 = np.concatenate([aero_tstep.forces[ss][0:3].reshape(-1, order='C')
                                  for ss in range(aero_tstep.n_surf)])

            # Assemble the state space system
            lin_uvlm_system.assemble_ss()
            self.data.aero.linear['System'] = lin_uvlm_system
            self.data.aero.linear['SS'] = lin_uvlm_system.SS
            self.data.aero.linear['x_0'] = x_0
            self.data.aero.linear['u_0'] = u_0
            self.data.aero.linear['f_0'] = f_0
            self.data.aero.linear['gamma_0'] = gamma
            self.data.aero.linear['gamma_star_0'] = gamma_star
            self.data.aero.linear['gamma_dot_0'] = gamma_dot
        # Initialise velocity generator
        velocity_generator_type = gen_interface.generator_from_string(self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'])


    def run(self,
            aero_tstep,
            structure_tstep,
            convect_wake=False,
            dt=None,
            t=None,
            unsteady_contribution=False):
        r"""
        Solve the linear aerodynamic UVLM model at the current time step. The already created linearised UVLM
        system is of the form:

        .. math::
            \mathbf{\dot{x}} &= \mathbf{A\,x} + \mathbf{B\,u} \\
            \mathbf{y} &= \mathbf{C\,x} + \mathbf{D\,u}


        The method does the following:

            1. Reshape SHARPy's data into column vector form for the linear UVLM system to use. SHARPy's data for the
            relevant inputs is stored in a list, with one entry per surface. Each surface entry is of shape
            ``(3, M+1, N+1)``, where 3 corresponds to ``x``,``y`` or ``z``, ``M`` number of chordwise panels and
            ``N`` number of spanwise panels.

                To reshape, for each surface the ``.reshape(-1, order='C')`` method is used, transforming the matrix
                into a column changing the last index the fastest.

                This is done for all 3 components of the input to the linear UVLM system:

                 * ``zeta``, :math:`\zeta`: panel grid coordinates
                 * ``zeta_dot``, :math:`\dot{\zeta}`: panel grid velocity
                 * ``u_ext``, :math:`u_{ext}`: external velocity


            2. Find variations with respect to initial reference state. The input to the linear UVLM system is of the
            form:

                .. math:: \mathbf{u} = [\delta\mathbf{\zeta}^T,\,
                    \delta\dot{\mathbf{\zeta}}^T,\,\delta\mathbf{u}^T_{ext}]^T

                Therefore, the values for panel coordinates and velocities at the reference are subtracted and
                concatenated to form a single column vector.

            3. The linear UVLM system is then solved using the method chosen by the user. The output is a column vector
            containing the aerodynamic forces at the panel vertices i.e. :math:`\mathbf{y} = [\mathbf{f}]`

            4. The vector of aerodynamic forces is then reshaped into the original SHARPy form following the reverse
            process of 1 and the ``forces`` field in ``self.data`` updated. Note: contrary to the shape of ``zeta``,
            ``forces`` consists of a ``(6, M+1, N+1)`` matrix for each surface but the bottom 3 rows are null.

        To Do:
            Introduce unsteady effects

        Args:
            aero_tstep (AeroTimeStepInfo): object containing the aerodynamic data at the current time step
            structure_tstep (StructTimeStepInfo): object containing the structural data at the current time step
            convect_wake (bool): for backward compatibility only. The linear UVLM assumes a frozen wake geometry
            dt (float): time increment
            t (float): current time
            unsteady_contribution (bool): include unsteady aerodynamic effects

        Returns:
            PreSharpy: updated ``self.data`` class with the new forces acting on the system

        """

        if aero_tstep is None:
            aero_tstep = self.data.aero.timestep_info[-1]
        if structure_tstep is None:
            structure_tstep = self.data.structure.timestep_info[-1]
        if dt is None:
            dt = self.settings['dt'].value
        if t is None:
            t = self.data.ts*dt

        if unsteady_contribution:
            raise NotImplementedError('Unsteady aerodynamic effects have not yet been implemented')

        # Generate external velocity field u_ext
        self.velocity_generator.generate({'zeta': aero_tstep.zeta,
                                          'override': True,
                                          't': t,
                                          'ts': self.data.ts,
                                          'dt': dt,
                                          'for_pos': structure_tstep.for_pos},
                                         aero_tstep.u_ext)

        # Reshape zeta, zeta_dot and u_ext into a column vector
        zeta = np.concatenate([aero_tstep.zeta[ss].reshape(-1, order='C')
                               for ss in range(aero_tstep.n_surf)])
        zeta_dot = np.concatenate([aero_tstep.zeta_dot[ss].reshape(-1, order='C')
                                   for ss in range(aero_tstep.n_surf)])
        u_ext = np.concatenate([aero_tstep.u_ext[ss].reshape(-1, order='C')
                               for ss in range(aero_tstep.n_surf)])

        # Column vector that will be the input to the linearised UVLM system
        # Variation between linearised state and perturbed state
        du_sta = np.concatenate((zeta, zeta_dot, u_ext)) - self.data.aero.linear['u_0']

        # Solve system - output is the variation in force
        dx_sta, dy_sta = self.data.aero.linear['System'].solve_steady(du_sta, method=self.settings['solution_method'])

        # Add reference value to obtain current state
        # Unpack state vector to update circulation terms
        dgamma_vec, dgamma_star_vec, dgamma_dot_vec = self.data.aero.linear['System'].unpack_state(dx_sta)
        gamma_vec = self.data.aero.linear['gamma_0'] + dgamma_vec
        gamma_star_vec = self.data.aero.linear['gamma_star_0'] + dgamma_star_vec
        gamma_dot_vec = self.data.aero.linear['gamma_dot_0'] + dgamma_dot_vec

        # Nodal forces column vector plus reference value
        f_aero = self.data.aero.linear['f_0'] + dy_sta

        # Reshape output into forces[i_surface] where forces[i_surface] is a (6,M+1,N+1) matrix
        forces = []
        gamma = []
        gamma_star = []
        gamma_dot = []

        worked_points = 0
        worked_panels = 0
        worked_wake_panels = 0

        for i_surf in range(aero_tstep.n_surf):
            # Tuple with dimensions of the aerogrid zeta, which is the same shape for forces
            dimensions = aero_tstep.zeta[i_surf].shape
            dimensions_gamma = self.data.aero.aero_dimensions[i_surf]
            dimensions_wake = self.data.aero.aero_dimensions_star[i_surf]

            # Number of entries in zeta
            points_in_surface = aero_tstep.zeta[i_surf].size
            panels_in_surface = aero_tstep.gamma[i_surf].size
            panels_in_wake = aero_tstep.gamma_star[i_surf].size

            # Append reshaped forces to each entry in list (one for each surface)
            forces.append(f_aero[worked_points:worked_points+points_in_surface].reshape(dimensions, order='C'))

            # Add the null bottom 3 rows to to the forces entry
            forces[i_surf] = np.concatenate((forces[i_surf], np.zeros(dimensions)))

            # Reshape bound circulation terms
            gamma.append(gamma_vec[worked_panels:worked_panels+panels_in_surface].reshape(
                dimensions_gamma, order='C'))
            gamma_dot.append(gamma_dot_vec[worked_panels:worked_panels+panels_in_surface].reshape(
                dimensions_gamma, order='C'))

            # Reshape wake circulation terms
            gamma_star.append(gamma_star_vec[worked_wake_panels:worked_wake_panels+panels_in_wake].reshape(
                dimensions_wake, order='C'))

            worked_points += points_in_surface
            worked_panels += panels_in_surface
            worked_wake_panels += panels_in_wake

        aero_tstep.forces = forces
        aero_tstep.gamma = gamma
        aero_tstep.gamma_dot = gamma_dot
        aero_tstep.gamma_star = gamma_star

        return self.data

    def add_step(self):
        self.data.aero.add_timestep()

    def update_grid(self, beam):
        self.data.aero.generate_zeta(beam, self.data.aero.aero_settings, -1, beam_ts=-1)

    def update_custom_grid(self, structure_tstep, aero_tstep):
        self.data.aero.generate_zeta_timestep_info(structure_tstep, aero_tstep, self.data.structure, self.data.aero.aero_settings)