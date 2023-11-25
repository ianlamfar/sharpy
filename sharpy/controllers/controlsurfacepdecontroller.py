import numpy as np

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.settings as settings
import sharpy.utils.control_utils as control_utils
import sharpy.utils.cout_utils as cout
import sharpy.aero.utils.mapping as mapping
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.utils as aeroutils


@controller_interface.controller
class ControlSurfacePdeController(controller_interface.BaseController):
    r"""


    """
    controller_id = 'ControlSurfacePdeController'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['time_history_input_file'] = 'str'
    settings_default['time_history_input_file'] = None
    settings_description['time_history_input_file'] = 'Route and file name of the time history of desired state'

    settings_types['P'] = 'float'
    settings_default['P'] = None
    settings_description['P'] = 'Proportional gain of the controller'

    settings_types['I'] = 'float'
    settings_default['I'] = 0.0
    settings_description['I'] = 'Integral gain of the controller'

    settings_types['D'] = 'float'
    settings_default['D'] = 0.0
    settings_description['D'] = 'Differential gain of the controller'

    # settings_types['input_type'] = 'str'
    # settings_default['input_type'] = 'lift'
    # settings_description['input_type'] = (
    #     'Quantity used to define the' +
    #     ' reference state. Supported: `pitch`')

    settings_types['dt'] = 'float'
    settings_default['dt'] = None
    settings_description['dt'] = 'Time step of the simulation'

    settings_types['controlled_surfaces'] = 'list(int)'
    settings_default['controlled_surfaces'] = None
    settings_description['controlled_surfaces'] = (
        'Control surface indices to be actuated by this controller')

    settings_types['controlled_surfaces_coeff'] = 'list(float)'
    settings_default['controlled_surfaces_coeff'] = [1.]
    settings_description['controlled_surfaces_coeff'] = (
        'Control surface deflection coefficients. ' +
        'For example, for antisymmetric deflections => [1, -1].')

    settings_types['write_controller_log'] = 'bool'
    settings_default['write_controller_log'] = True
    settings_description['write_controller_log'] = (
        'Write a time history of input, required input, ' +
        'and control')

    settings_types['controller_log_route'] = 'str'
    settings_default['controller_log_route'] = './output/'
    settings_description['controller_log_route'] = (
        'Directory where the log will be stored')

    settings_types['controller_noise'] = 'bool'
    settings_default['controller_noise'] = False
    settings_description['controller_noise'] = 'adding white noise to controller output'

    settings_types['controller_noise_settings'] = 'dict'
    settings_default['controller_noise_settings'] = dict()
    settings_description['controller_noise_settings'] = 'settings for controller white noise'

    settings_types['controller_lag'] = 'int'
    settings_default['controller_lag'] = 0
    settings_description['controller_lag'] = 'lag of the controller input, defined in timesteps (default is 0)'

    settings_types['output_limit'] = 'float'
    settings_default['output_limit'] = -1
    settings_description['output_limit'] = 'capping the controller output to a defined range -limit <= output <= limit, -1 if no limit'
    
    settings_types['rho'] = 'float'
    settings_default['rho'] = 1.225
    settings_description['rho'] = 'density for CL calculation'


    # supported_input_types = ['pitch', 'roll', 'pos_', 'tip rotation', 'lift']

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types,
                                       settings_default,
                                       settings_description)

    def __init__(self):
        self.in_dict = None
        self.data = None
        self.settings = None

        self.prescribed_input_time_history = None

        # Time histories are ordered such that the [i]th element of each
        # is the state of the controller at the time of returning.
        # That means that for the timestep i,
        # state_input_history[i] == input_time_history_file[i] + error[i]
        self.p_error_history = list()
        self.i_error_history = list()
        # self.d_error_history = list()
        self.real_state_input_history = list()
        self.control_history = list()

        self.controller_implementation = None

        self.n_control_surface = 0

        self.log = None

    def initialise(self, data, in_dict, controller_id=None, restart=False):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default)

        self.settings = self.in_dict
        self.controller_id = controller_id

        # validate that the input_type is in the supported ones
        # valid = False
        # for t in self.supported_input_types:
        #     if t in self.settings['input_type']:
        #         valid = True
        #         break
        # if not valid:
        #     cout.cout_wrap('The input_type {} is not supported by {}'.format(
        #         self.settings['input_type'], self.controller_id), 3)
        #     cout.cout_wrap('The supported ones are:', 3)
        #     for i in self.supported_input_types:
        #         cout.cout_wrap('    {}'.format(i), 3)
        #     raise NotImplementedError()

        if self.settings['write_controller_log']:
            self.log = open(self.settings['controller_log_route'] + self.controller_id + '_log.csv', 'w+')
            self.log.write(('#' + 1 * '{:>2},' + 9 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'noise', 'capping', 'raw', 'control'))
            self.log.flush()

        # save input time history
        try:
            self.prescribed_input_time_history = (
                np.loadtxt(self.settings['time_history_input_file'], delimiter=','))
        except OSError:
            raise OSError('File {} not found in Controller'.format(self.settings['time_history_input_file']))

        # Init PID controller
        self.controller_implementation = control_utils.PDE(self.settings['P'],
                                                           self.settings['I'],
                                                           self.settings['D'],
                                                           self.settings['dt'])

        # check that controlled_surfaces_coeff has the correct number of parameters
        # if len() == 1 and == 1.0, then expand to number of surfaces.
        # if len(coeff) /= n_surfaces, throw error
        self.n_control_surface = len(self.settings['controlled_surfaces'])
        if (len(self.settings['controlled_surfaces_coeff']) ==
                self.n_control_surface):
            # All good, pass checks
            pass
        elif (len(self.settings['controlled_surfaces_coeff']) == 1 and
              self.settings['controlled_surfaces_coeff'][0] == 1.0):
            # default value, fill with 1.0
            self.settings['controlled_surfaces_coeff'] = np.ones(
                (self.n_control_surface,), dtype=float)
        else:
            raise ValueError('controlled_surfaces_coeff does not have as many'
                             + ' elements as controller_surfaces')

        if self.settings['controller_lag'] < 0:
            raise ValueError('negative contorller_lag is not allowed')
        elif type(self.settings['controller_lag']) != int:
            raise TypeError('controller_lag is not type (int)')

    def control(self, data, controlled_state):
        r"""
        Main routine of the controller.
        Input is `data` (the self.data in the solver), and
        `currrent_state` which is a dictionary with ['structural', 'aero']
        time steps for the current iteration.

        :param data: problem data containing all the information.
        :param controlled_state: `dict` with two vars: `structural` and `aero`
            containing the `timestep_info` that will be returned with the
            control variables.

        :returns: A `dict` with `structural` and `aero` time steps and control
            input included.
        """
        self.data = data
        # get current state input
        self.real_state_input_history.append(self.extract_time_history(controlled_state))
        # print(self.real_state_input_history)

        # apply lag to state, DO NOT ALTER "data" or "controlled_state" as it contains other info of current timestep
        lag = self.settings['controller_lag']
        lag_index = len(self.real_state_input_history)
        if lag > 1:  # 0=default, 1=effectively default (previous timestep)
            lag_index = max(0, len(self.real_state_input_history) - lag) + 1

        i_current = len(self.real_state_input_history)
        # print(len(self.real_state_input_history), np.degrees(self.real_state_input_history[lag_index - 1]), i_current, lag_index)
        # apply it where needed.
        control_command, detail = self.controller_wrapper(
            required_input=self.prescribed_input_time_history,
            current_input=self.real_state_input_history,
            control_param={'P': self.settings['P'],
                           'I': self.settings['I'],
                           'D': self.settings['D'],
                           },
            i_current=i_current,
            lag_index=lag_index)
        
        # print(control_command, detail)

        # save raw command before applying noise and cap
        raw_command = control_command

        # adding white noise
        noise = 0
        if self.settings['controller_noise']:
            noise_settings = self.settings['controller_noise_settings']
            noise_mode = noise_settings['noise_mode']  # either max percentage (multiply to signal) or max amplitude (add to signal)
            noise_percentage = float(noise_settings['max_percentage'])  # max percentage of noise with respect to signal at that timestep
            noise_amplitude = float(noise_settings['max_amplitude'])  # max amplitude of noise
            if noise_percentage > 1:  # divide by 100 for values in percentage scale
                noise_percentage /= 100

            if noise_mode == 'percentage':
                noise = np.random.normal(0, noise_percentage)
                control_command *= (1 + noise)  # multiply the output by the noise
            elif noise_mode == 'amplitude':
                noise = np.random.normal(0, noise_amplitude)
                control_command += noise  # add noise to output

        # apply controller output cap
        output_cap = self.settings['output_limit']
        cap = 0
        if output_cap != -1 and abs(control_command) >= abs(output_cap):
            # if control_command < 0:
            #     control_command = - abs(output_cap)
            #     cap = -1
            # elif control_command > 0:
            #     control_command = abs(output_cap)
            #     cap = 1
            cap = np.sign(control_command)
            control_command = cap * abs(output_cap)

        controlled_state['aero'].control_surface_deflection = (
            np.array(self.settings['controlled_surfaces_coeff']) * control_command)

        self.log.write(('{:>6d},' + 9 * '{:>12.6f},' + '{:>12.6f}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              self.prescribed_input_time_history[i_current - 1],
                              self.real_state_input_history[i_current - 1],
                              detail[0],
                              detail[1],
                              detail[2],
                              noise,
                              cap,
                              raw_command,
                              control_command))
        self.log.flush()
        error = self.prescribed_input_time_history[i_current - 1]-self.real_state_input_history[i_current - 1]
        raw = np.degrees(raw_command)
        control = np.degrees(detail)
        cap_control = np.degrees(control_command)
        print(f'PDEControl -- error: {error:+.3f}, raw: {raw:+.3f} [{control[0]:+.3f}P|{control[1]:+.3f}I|{control[2]:+.3f}D], capped: {cap_control:+.3f}')
        # print(controlled_state['structural'].psi[-1, -1, 1], controlled_state['aero'].control_surface_deflection)
        return controlled_state

    def extract_time_history(self, controlled_state):
        output: float = 0.0
        aero_tstep = controlled_state['aero']
        struct_tstep = controlled_state['structural']
        lift = 0
        output = np.sum(lift)
        
        forces = mapping.aero2struct_force_mapping(
            aero_tstep.forces + aero_tstep.dynamic_forces,
            self.data.aero.struct2aero_mapping,
            aero_tstep.zeta,
            struct_tstep.pos,
            struct_tstep.psi,
            self.data.structure.node_master_elem,
            self.data.structure.connectivities,
            struct_tstep.cag(),
            self.data.aero.aero_dict)
        
        N_nodes = self.data.structure.num_node
        numb_col = 4
        header = "x,y,z,fz"
        # get aero forces
        lift_distribution = np.zeros((N_nodes, numb_col))
        # get rotation matrix
        cga = algebra.quat2rotation(struct_tstep.quat)
        
        header += ", y/s, cl"
        numb_col += 2
        lift_distribution = np.concatenate((lift_distribution, np.zeros((N_nodes, 2))), axis=1)

        total_area = 0
        for inode in range(N_nodes):
            if self.data.aero.aero_dict['aero_node'][inode]:
                local_node = self.data.aero.struct2aero_mapping[inode][0]["i_n"]
                ielem, inode_in_elem = self.data.structure.node_master_elem[inode]
                i_surf = int(self.data.aero.surface_distribution[ielem])
                # get c_gb
                cab = algebra.crv2rotation(struct_tstep.psi[ielem, inode_in_elem, :])
                cgb = np.dot(cga, cab)
                # Get c_bs
                urel, dir_urel = aeroutils.magnitude_and_direction_of_relative_velocity(struct_tstep.pos[inode, :],
                                                                                        struct_tstep.pos_dot[inode, :],
                                                                                        struct_tstep.for_vel[:],
                                                                                        cga,
                                                                                        aero_tstep.u_ext[i_surf][:, :,
                                                                                        local_node])
                dir_span, span, dir_chord, chord = aeroutils.span_chord(local_node, aero_tstep.zeta[i_surf])
                total_area += span * chord
                # Stability axes - projects forces in B onto S
                c_bs = aeroutils.local_stability_axes(cgb.T.dot(dir_urel), cgb.T.dot(dir_chord))
                lift_force = c_bs.T.dot(forces[inode, :3])[2]
                # Store data in export matrix
                lift_distribution[inode, 3] = lift_force
                lift_distribution[inode, 2] = struct_tstep.pos[inode, 2]  # z
                lift_distribution[inode, 1] = struct_tstep.pos[inode, 1]  # y
                lift_distribution[inode, 0] = struct_tstep.pos[inode, 0]  # x
                
                lift_distribution[inode, 4] = lift_distribution[inode, 1]/span
                # Get lift coefficient
                u = np.linalg.norm(urel)
                if u > 0:
                    lift_distribution[inode, 5] = (np.sign(lift_force) *
                                                   np.linalg.norm(lift_force) /
                                                   (0.5 * self.settings['rho'] *
                                                    (u ** 2) * span * chord))  # strip_area[i_surf][local_node])
                else:
                    lift_distribution[inode, 5] = 0
                # Check if shared nodes from different surfaces exist (e.g. two wings joining at symmetry plane)
                # Leads to error since panel area just donates for half the panel size while lift forces is summed up
                lift_distribution[inode, 5] /= len(self.data.aero.struct2aero_mapping[inode])
        
        
        lift = np.sum(lift_distribution[:, 3])  # total lift force
        # u = np.linalg.norm(urel)
        # if u > 0.0:
        #     output = lift / (0.5 * self.settings['rho'] * (u**2) * total_area)
        # else: output = 0.0
        
        output = np.mean(lift_distribution[:, 5])
        # print(lift_distribution[:, -1], output, np.mean(lift_distribution))
        # output = np.sqrt(np.abs(output)) * np.sign(output)

        return output

    def controller_wrapper(self,
                           required_input,
                           current_input,
                           control_param,
                           i_current,
                           lag_index):
        self.controller_implementation.set_point(required_input[i_current - 1])
        control_param, detailed_control_param = self.controller_implementation(current_input[lag_index - 1])
        return (control_param, detailed_control_param)

    def __exit__(self, *args):
        self.log.close()
