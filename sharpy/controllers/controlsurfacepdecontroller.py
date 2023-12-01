import numpy as np
import scipy.signal as sig
import scipy

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

    settings_types['LP'] = 'float'
    settings_default['LP'] = 0.0
    settings_description['LP'] = 'Proportional gain of the controller'

    settings_types['LI'] = 'float'
    settings_default['LI'] = 0.0
    settings_description['LI'] = 'Integral gain of the controller'

    settings_types['LD'] = 'float'
    settings_default['LD'] = 0.0
    settings_description['LD'] = 'First differential gain of the controller'
    
    settings_types['LD2'] = 'float'
    settings_default['LD2'] = 0.0
    settings_description['LD2'] = 'Second differential gain of the controller'

    settings_types['HP'] = 'float'
    settings_default['HP'] = 0.0
    settings_description['HP'] = 'Proportional gain of the controller'

    settings_types['HI'] = 'float'
    settings_default['HI'] = 0.0
    settings_description['HI'] = 'Integral gain of the controller'

    settings_types['HD'] = 'float'
    settings_default['HD'] = 0.0
    settings_description['HD'] = 'First differential gain of the controller'
    
    settings_types['HD2'] = 'float'
    settings_default['HD2'] = 0.0
    settings_description['HD2'] = 'Second differential gain of the controller'
    
    settings_types['P_rampup_steps'] = 'int'
    settings_default['P_rampup_steps'] = 1
    
    settings_types['I_rampup_steps'] = 'int'
    settings_default['I_rampup_steps'] = 1
    
    settings_types['D_rampup_steps'] = 'int'
    settings_default['D_rampup_steps'] = 1
    
    settings_types['D2_rampup_steps'] = 'int'
    settings_default['D2_rampup_steps'] = 1
    
    settings_types['order'] = 'int'
    settings_default['order'] = 2
    settings_description['order'] = 'finite difference order for derivative'
    
    settings_types['cutoff_freq'] = 'float'
    settings_default['cutoff_freq'] = 30

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

    settings_types['error_pow'] = 'float'
    settings_default['error_pow'] = 1.0

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
            self.log_lp = open(self.settings['controller_log_route'] + self.controller_id + '_lp_log.csv', 'w+')
            self.log_lp.write(('#' + 1 * '{:>2},' + 10 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'D2control', 'noise', 'capping', 'raw', 'control'))
            self.log_lp.flush()
            
            self.log_hp = open(self.settings['controller_log_route'] + self.controller_id + '_hp_log.csv', 'w+')
            self.log_hp.write(('#' + 1 * '{:>2},' + 10 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'D2control', 'noise', 'capping', 'raw', 'control'))
            self.log_hp.flush()

        # save input time history
        try:
            self.prescribed_input_time_history = (
                np.loadtxt(self.settings['time_history_input_file'], delimiter=','))
        except OSError:
            raise OSError('File {} not found in Controller'.format(self.settings['time_history_input_file']))

        # Init PID controller
        self.controller_implementation_lp = control_utils.PDE(self.settings['LP'],
                                                           self.settings['LI'],
                                                           self.settings['LD'],
                                                           self.settings['LD2'],
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        
        self.controller_implementation_hp = control_utils.PDE(self.settings['HP'],
                                                           self.settings['HI'],
                                                           self.settings['HD'],
                                                           self.settings['HD2'],
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )

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
        
        lp_input, hp_input = self.filter(self.real_state_input_history)

        # lp_input = self.real_state_input_history - hp_input
        
        
        # # hp_input = self.filter(self.real_state_input_history, 'hp')
        # # lp_input = self.real_state_input_history - hp_input
        
        control_command_lp, detail_lp = self.controller_wrapper(
            required_input=self.prescribed_input_time_history,
            current_input=lp_input,
            control_param={'P': self.settings['LP'],
                           'I': self.settings['LI'],
                           'D': self.settings['LD'],
                           'D2': self.settings['LD2'],
                           'P_steps': self.settings['P_rampup_steps'],
                           'I_steps': self.settings['I_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='low',
            )
        
        control_command_hp, detail_hp = self.controller_wrapper(
            required_input=np.zeros_like(self.prescribed_input_time_history),
            current_input=hp_input,
            control_param={'P': self.settings['HP'],
                           'I': self.settings['HI'],
                           'D': self.settings['HD'],
                           'D2': self.settings['HD2'],
                           'P_steps': self.settings['P_rampup_steps'],
                           'I_steps': self.settings['I_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='high',
            )
        
        # print(control_command, detail)

        # save raw command before applying noise and cap
        # abs_err = abs(self.prescribed_input_time_history[-1] - self.real_state_input_history[-1])
        # if abs_err > 0:
        #     hp_weight = abs(hp_input[-1]) / abs_err
        #     lp_weight = 1 - hp_weight
        # else:
        #     hp_weight = 0.5
        #     lp_weight = 0.5
        
        hp_weight = 1
        lp_weight = 1
        
        control_command_hp *= hp_weight
        control_command_lp *= lp_weight
        detail_hp *= hp_weight
        detail_lp *= lp_weight
        
        control_command = control_command_lp + control_command_hp
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
            cap = np.sign(control_command)
            control_command = cap * abs(output_cap)

        controlled_state['aero'].control_surface_deflection = (
            np.array(self.settings['controlled_surfaces_coeff']) * control_command)

        self.log_lp.write(('{:>6d},' + 10 * '{:>12.6f},' + '{:>12.6f}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              self.prescribed_input_time_history[i_current - 1],
                              lp_input[i_current - 1],
                              detail_lp[0],
                              detail_lp[1],
                              detail_lp[2],
                              detail_lp[3],
                              noise,
                              cap,
                              raw_command,
                              control_command_lp))
        self.log_lp.flush()
        
        self.log_hp.write(('{:>6d},' + 10 * '{:>12.6f},' + '{:>12.6f}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              np.zeros_like(self.prescribed_input_time_history[i_current - 1]),
                              hp_input[i_current - 1],
                              detail_hp[0],
                              detail_hp[1],
                              detail_hp[2],
                              detail_hp[3],
                              noise,
                              cap,
                              raw_command,
                              control_command_hp))
        self.log_hp.flush()
        
        
        
        error_hp = -hp_input[i_current - 1]
        error_lp = self.prescribed_input_time_history[i_current - 1] - lp_input[i_current - 1]
        
        # error = self.prescribed_input_time_history[i_current - 1] - self.real_state_input_history[i_current - 1]
        error = error_hp + error_lp
        
        raw = np.degrees(raw_command)
        # control = np.degrees(detail)
        control_lp = np.degrees(detail_lp)
        control_hp = np.degrees(detail_hp)
        cap_control = np.degrees(control_command)
        print(f'PDEControl -- error: {error:+.4e}, raw: {raw:+.4f}, capped: {cap_control:+.4f}')
        print(f'LP -- error: {error_lp:+.4e}, [{control_lp[0]:+.4f}P|{control_lp[1]:+.4f}I|{control_lp[2]:+.4f}D|{control_lp[3]:+.4f}D2]')
        print(f'HP -- error: {error_hp:+.4e}, [{control_hp[0]:+.4f}P|{control_hp[1]:+.4f}I|{control_hp[2]:+.4f}D|{control_hp[3]:+.4f}D2]')
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
        # print(lift_distribution[:, -1].shape, (self.data.aero.aero_dict['control_surface'] != 0).shape)
        # output = np.sqrt(np.abs(output)) * np.sign(output)

        return output

    def controller_wrapper(self,
                           required_input,
                           current_input,
                           control_param,
                           i_current,
                           lag_index,
                           mode,
                           ):
        order = 10
        freq = 30
        if mode == 'low' or mode == 'lp':
            controller = self.controller_implementation_lp
        elif mode == 'high' or mode == 'hp':
            controller = self.controller_implementation_hp
        else:
            raise NotImplementedError('controller filter mode is either low/lp or high/hp')
        
        controller.set_point(required_input[i_current - 1])        
        control_param, detailed_control_param = controller(current_input[lag_index - 1],
                                                            power=self.settings['error_pow'],
                                                            rampups=[control_param['P_steps'],
                                                                    control_param['I_steps'],
                                                                    control_param['D_steps'],
                                                                    control_param['D2_steps'],
                                                                    ]
                                                            )
        return (control_param, detailed_control_param)
    
    
    def filter(self, data, mode=None, order=10, freq=30):
        # if mode == 'low' or mode == 'lp':
        #     b, a = sig.butter(order, freq, btype='lowpass', analog=False, fs=(1/self.settings['dt']))
        # elif mode == 'high' or mode == 'hp':
        #     b, a = sig.butter(order, freq, btype='highpass', analog=False, fs=(1/self.settings['dt']))
        # try:
        #     data = sig.filtfilt(b, a, data)
        # except:
        #     data = sig.lfilter(b, a, data)
        
        data_fft = scipy.fft.fft(data, n=len(data))
        fft_freq = scipy.fft.fftfreq(data_fft.shape[0], d=self.settings['dt'])
        
        data_lp = data_fft.copy()
        data_hp = data_fft.copy()
        
        
        data_lp[np.abs(fft_freq) > self.settings['cutoff_freq']] = 0
        data_hp[np.abs(fft_freq) <= self.settings['cutoff_freq']] = 0
        
        data_lp = scipy.fft.ifft(data_lp, n=data_fft.shape[0])
        data_hp = scipy.fft.ifft(data_hp, n=data_fft.shape[0])
        
        return data_lp.real, data_hp.real

    def __exit__(self, *args):
        self.log.close()
