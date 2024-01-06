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
    
    settings_types['Kpos'] = 'float'
    settings_default['Kpos'] = 0

    ############ CONTROLLER CONSTANTS ############

    settings_types['P'] = 'list(float)'
    settings_default['P'] = [0, 0]
    settings_description['P'] = 'Proportional gain of the controller'

    settings_types['I'] = 'list(float)'
    settings_default['I'] = [0, 0]
    settings_description['I'] = 'Integral gain of the controller'

    settings_types['D'] = 'list(float)'
    settings_default['D'] = [0, 0]
    settings_description['D'] = 'First differential gain of the controller'
    
    settings_types['D2'] = 'list(float)'
    settings_default['D2'] = [0, 0]
    settings_description['D2'] = 'Second differential gain of the controller'

    ############ CONTROLLER CONSTANTS ############
    
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
    
    settings_types['cutoff_freq0'] = 'list(float)'
    settings_default['cutoff_freq0'] = [0, 5]
    
    settings_types['cutoff_freq1'] = 'list(float)'
    settings_default['cutoff_freq1'] = [32, 40]
    
    settings_types['smoothing'] = 'bool'
    settings_default['smoothing'] = True
    
    settings_types['kernel_size'] = 'int'
    settings_default['kernel_size'] = 3
    
    settings_types['kernel_mode'] = 'str'
    settings_default['kernel_mode'] = 'gaussian'

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
        self.d_error_history = list()
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
        
        self.lp_active = (self.settings['P'][0]!=0 or self.settings['I'][0]!=0 or self.settings['D'][0]!=0 or self.settings['D2'][0]!=0)
        self.hp_active = (self.settings['P'][1]!=0 or self.settings['I'][1]!=0 or self.settings['D'][1]!=0 or self.settings['D2'][1]!=0)
        self.t_active = (self.settings['P'][2]!=0 or self.settings['I'][2]!=0 or self.settings['D'][2]!=0 or self.settings['D2'][2]!=0)
        self.td_active = (self.settings['P'][3]!=0 or self.settings['I'][3]!=0 or self.settings['D'][3]!=0 or self.settings['D2'][3]!=0)
        self.tdd_active = (self.settings['P'][4]!=0 or self.settings['I'][4]!=0 or self.settings['D'][4]!=0 or self.settings['D2'][4]!=0)

        if self.settings['write_controller_log']:
            self.log_tipr = open(self.settings['controller_log_route'] + self.controller_id + '_tip_r_log.csv', 'w+')
            self.log_tipr.write(('#' + 1 * '{:>2},' + 10 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'D2control', 'noise', 'capping', 'raw', 'control'))
            self.log_tipr.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(0,0,0,0,0,0,0,0,0,0,0,0))
            self.log_tipr.flush()
            
            self.log_tipz = open(self.settings['controller_log_route'] + self.controller_id + '_tip_z_log.csv', 'w+')
            self.log_tipz.write(('#' + 1 * '{:>2},' + 10 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'D2control', 'noise', 'capping', 'raw', 'control'))
            self.log_tipz.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(0,0,0,0,0,0,0,0,0,0,0,0))
            self.log_tipz.flush()
            
            self.log_t = open(self.settings['controller_log_route'] + self.controller_id + '_t_log.csv', 'w+')
            self.log_t.write(('#' + 1 * '{:>2},' + 10 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'D2control', 'noise', 'capping', 'raw', 'control'))
            self.log_t.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(0,0,0,0,0,0,0,0,0,0,0,0))
            self.log_t.flush()
            
            self.log_td = open(self.settings['controller_log_route'] + self.controller_id + '_td_log.csv', 'w+')
            self.log_td.write(('#' + 1 * '{:>2},' + 10 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'D2control', 'noise', 'capping', 'raw', 'control'))
            self.log_td.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(0,0,0,0,0,0,0,0,0,0,0,0))
            self.log_td.flush()
            
            self.log_tdd = open(self.settings['controller_log_route'] + self.controller_id + '_tdd_log.csv', 'w+')
            self.log_tdd.write(('#' + 1 * '{:>2},' + 10 * '{:>12},' + '{:>12}\n').
                           format('tstep', 'time', 'Ref. state', 'state', 'Pcontrol', 'Icontrol', 'Dcontrol', 'D2control', 'noise', 'capping', 'raw', 'control'))
            self.log_tdd.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(0,0,0,0,0,0,0,0,0,0,0,0))
            self.log_tdd.flush()

        # save input time history
        try:
            self.prescribed_input_time_history = (
                np.loadtxt(self.settings['time_history_input_file'], delimiter=','))
        except OSError:
            raise OSError('File {} not found in Controller'.format(self.settings['time_history_input_file']))

        # Init PID controller
        self.controller_implementation_tip_theta = control_utils.PDE(self.settings['P'][0],
                                                           self.settings['I'][0],
                                                           0,
                                                           0,
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        self.tip_theta_input_history = list()
        
        self.controller_implementation_tip_theta_dot = control_utils.PDE(self.settings['D'][0],
                                                           0,
                                                           0,
                                                           0,
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        self.tip_theta_dot_input_history = list()
        
        self.controller_implementation_tip_theta_ddot = control_utils.PDE(self.settings['D2'][0],
                                                           0,
                                                           0,
                                                           0,
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        self.tip_theta_ddot_input_history = list()
        
        self.controller_implementation_tip_pos = control_utils.PDE(self.settings['P'][1],
                                                           self.settings['I'][1],
                                                           self.settings['D'][1],
                                                           self.settings['D2'][1],
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        self.tip_pos_input_history = list()
        
        self.controller_implementation_theta = control_utils.PDE(self.settings['P'][2],
                                                           self.settings['I'][2],
                                                           self.settings['D'][2],
                                                           self.settings['D2'][2],
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        self.theta_input_history = list()
        
        self.controller_implementation_theta_dot = control_utils.PDE(self.settings['P'][3],
                                                           self.settings['I'][3],
                                                           self.settings['D'][3],
                                                           self.settings['D2'][3],
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        self.theta_dot_input_history = list()
        
        self.controller_implementation_theta_ddot = control_utils.PDE(self.settings['P'][4],
                                                           self.settings['I'][4],
                                                           self.settings['D'][4],
                                                           self.settings['D2'][4],
                                                           self.settings['dt'],
                                                           self.settings['order'],
                                                           )
        self.theta_ddot_input_history = list()

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
        self.tip_theta_input_history.append(self.extract_time_history(controlled_state, mode='tip_theta'))
        self.tip_theta_dot_input_history.append(self.extract_time_history(controlled_state, mode='tip_theta_dot'))
        self.tip_theta_ddot_input_history.append(self.extract_time_history(controlled_state, mode='tip_theta_ddot'))
        self.tip_pos_input_history.append(self.extract_time_history(controlled_state, mode='tip_pos'))
        self.theta_input_history.append(self.extract_time_history(controlled_state, mode='theta'))
        self.theta_dot_input_history.append(self.extract_time_history(controlled_state, mode='theta_dot'))
        self.theta_ddot_input_history.append(self.extract_time_history(controlled_state, mode='theta_ddot'))

        # apply lag to state, DO NOT ALTER "data" or "controlled_state" as it contains other info of current timestep
        lag = self.settings['controller_lag']
        lag_index = len(self.theta_input_history)
        if lag > 1:  # 0=default, 1=effectively default (previous timestep)
            lag_index = max(0, len(self.theta_input_history) - lag) + 1

        i_current = len(self.theta_input_history)
        # apply it where needed.
        
        control_command_tipr, detail_tipr = self.controller_wrapper(
            required_input=self.prescribed_input_time_history,
            current_input=self.tip_theta_input_history,
            control_param={'P_steps': self.settings['P_rampup_steps'],
                           'I_steps': self.settings['I_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='tip_theta',
            )
        
        _control_command_tiprd, _detail_tiprd = self.controller_wrapper(
            required_input=self.prescribed_input_time_history,
            current_input=self.tip_theta_dot_input_history,
            control_param={'P_steps': self.settings['D_rampup_steps'],
                           'I_steps': self.settings['D_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='tip_theta_dot',
            )
        
        _control_command_tiprdd, _detail_tiprdd = self.controller_wrapper(
            required_input=self.prescribed_input_time_history,
            current_input=self.tip_theta_ddot_input_history,
            control_param={'P_steps': self.settings['D2_rampup_steps'],
                           'I_steps': self.settings['D2_rampup_steps'],
                           'D_steps': self.settings['D2_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='tip_theta_ddot',
            )
        
        control_command_tipr += _control_command_tiprd + _control_command_tiprdd
        detail_tipr[-2] = _detail_tiprd[0]
        detail_tipr[-1] = _detail_tiprdd[0]
        
        control_command_tipz, detail_tipz = self.controller_wrapper(
            required_input=np.zeros_like(self.prescribed_input_time_history),
            current_input=self.tip_pos_input_history,
            control_param={'P_steps': self.settings['P_rampup_steps'],
                           'I_steps': self.settings['I_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='tip_pos',
            )
        
        control_command_t, detail_t = self.controller_wrapper(
            required_input=np.zeros_like(self.prescribed_input_time_history),
            current_input=self.theta_input_history,
            control_param={'P_steps': self.settings['P_rampup_steps'],
                           'I_steps': self.settings['I_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='theta',
            )
        
        control_command_td, detail_td = self.controller_wrapper(
            required_input=np.zeros_like(self.prescribed_input_time_history),
            current_input=self.theta_dot_input_history,
            control_param={'P_steps': self.settings['P_rampup_steps'],
                           'I_steps': self.settings['I_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='theta_dot',
            )
        
        control_command_tdd, detail_tdd = self.controller_wrapper(
            required_input=np.zeros_like(self.prescribed_input_time_history),
            current_input=self.theta_ddot_input_history,
            control_param={'P_steps': self.settings['P_rampup_steps'],
                           'I_steps': self.settings['I_rampup_steps'],
                           'D_steps': self.settings['D_rampup_steps'],
                           'D2_steps': self.settings['D2_rampup_steps'],
                           },
            i_current=i_current,
            lag_index=lag_index,
            mode='theta_ddot',
            )
               
        control_command = control_command_tipr + control_command_tipz +\
            control_command_t + control_command_td + control_command_tdd
        raw_command = control_command

        # adding white noise
        noise = 0
        if self.settings['controller_noise']:
            noise_settings = self.settings['controller_noise_settings']
            noise_mode = noise_settings['noise_mode']  # either max percentage (multiply to signal) or max amplitude (add to signal)
            noise_percentage = float(noise_settings['max_percentage'])  # max percentage of noise with respect to signal at that timestep
            noise_amplitude = float(noise_settings['max_amplitude'])  # max amplitude of noise
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

        self.log_tipr.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              self.prescribed_input_time_history[i_current - 1],
                              self.tip_theta_input_history[i_current - 1],
                              detail_tipr[0],
                              detail_tipr[1],
                              detail_tipr[2],
                              detail_tipr[3],
                              noise,
                              cap,
                              raw_command,
                              control_command_tipr))
        self.log_tipr.flush()
        
        self.log_tipz.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              np.zeros_like(self.prescribed_input_time_history[i_current - 1]),
                              self.tip_pos_input_history[i_current - 1],
                              detail_tipz[0],
                              detail_tipz[1],
                              detail_tipz[2],
                              detail_tipz[3],
                              noise,
                              cap,
                              raw_command,
                              control_command_tipz))
        self.log_tipz.flush()
        
        self.log_t.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              np.zeros_like(self.prescribed_input_time_history[i_current - 1]),
                              self.theta_input_history[i_current - 1],
                              detail_t[0],
                              detail_t[1],
                              detail_t[2],
                              detail_t[3],
                              noise,
                              cap,
                              raw_command,
                              control_command_t))
        self.log_t.flush()
        
        self.log_td.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              np.zeros_like(self.prescribed_input_time_history[i_current - 1]),
                              self.theta_dot_input_history[i_current - 1],
                              detail_td[0],
                              detail_td[1],
                              detail_td[2],
                              detail_td[3],
                              noise,
                              cap,
                              raw_command,
                              control_command_td))
        self.log_td.flush()
        
        self.log_tdd.write(('{:>6d},' + 10 * '{:>12.6e},' + '{:>12.6e}\n').
                       format(i_current,
                              i_current * self.settings['dt'],
                              np.zeros_like(self.prescribed_input_time_history[i_current - 1]),
                              self.theta_ddot_input_history[i_current - 1],
                              detail_tdd[0],
                              detail_tdd[1],
                              detail_tdd[2],
                              detail_tdd[3],
                              noise,
                              cap,
                              raw_command,
                              control_command_tdd))
        self.log_tdd.flush()
        
        
        
        error_tipr = -self.tip_theta_input_history[i_current - 1]
        error_tipz = self.prescribed_input_time_history[i_current - 1] - self.tip_pos_input_history[i_current - 1]
        error_t = -self.theta_input_history[i_current - 1]
        error_td = -self.theta_dot_input_history[i_current - 1]
        error_tdd = -self.theta_ddot_input_history[i_current - 1]
        
        
        error_tipr *= self.lp_active
        error_tipz *= self.hp_active
        error_t *= self.t_active
        error_td *= self.td_active
        error_tdd *= self.tdd_active
        
        error = error_tipr + error_tipz + error_t
        
        raw = np.degrees(raw_command)
        # control = np.degrees(detail)
        control_tipr = np.degrees(detail_tipr)
        control_tipz = np.degrees(detail_tipz)
        control_t = np.degrees(detail_t)
        control_td = np.degrees(detail_td)
        control_tdd = np.degrees(detail_tdd)
        cap_control = np.degrees(control_command)
        
        cmd_tipr = np.degrees(control_command_tipr)
        cmd_tipz = np.degrees(control_command_tipz)
        cmd_t = np.degrees(control_command_t + control_command_td + control_command_tdd)
        
        print(f'PDEControl -- error: {error:+.4e}, raw: {raw:+.4f}, capped: {cap_control:+.4f}')
        if self.lp_active: print(f'θt -- error: {error_tipr:+.4e}, [{cmd_tipr:+.4f}Σ||{control_tipr[0]:+.4f}P|{control_tipr[1]:+.4f}I|{control_tipr[2]:+.4f}D|{control_tipr[3]:+.4f}D²]')
        if self.hp_active: print(f'zt -- error: {error_tipz:+.4e}, [{cmd_tipz:+.4f}Σ||{control_tipz[0]:+.4f}P|{control_tipz[1]:+.4f}I|{control_tipz[2]:+.4f}D|{control_tipz[3]:+.4f}D²]')
        if self.t_active: print(f'∫θ -- error: {error_t:+.4e}, [{cmd_t:+.4f}Σ||{control_t[0]:+.4f}P|{control_t[1]:+.4f}I|{control_td[0]:+.4f}D|{control_tdd[0]:+.4f}D²|{control_tdd[2]:+.4f}D³|{control_tdd[3]:+.4f}D⁴]')
        # if self.td_active: print(f'T1 -- error: {error_td:+.4e}, [{control_td[0]:+.4f}P|{control_td[1]:+.4f}I|{control_td[2]:+.4f}D|{control_td[3]:+.4f}D2]')
        # if self.tdd_active: print(f'T2 -- error: {error_tdd:+.4e}, [{control_tdd[0]:+.4f}P|{control_tdd[1]:+.4f}I|{control_tdd[2]:+.4f}D|{control_tdd[3]:+.4f}D2]')
        
        return controlled_state

    def extract_time_history(self, controlled_state, mode):
        output: float = 0.0
        aero_tstep = controlled_state['aero']
        struct_tstep = controlled_state['structural']
        lift = 0
        output = np.sum(lift)
        
        not_cs = [np.expand_dims((i != 1), 0) for i in self.data.aero.cs_nodes]
        
        forces_excl_cs = [aero_tstep.forces[i] * not_cs[i] for i in range(aero_tstep.n_surf)]
        force_diff = [aero_tstep.forces[i] - forces_excl_cs[i] for i in range(aero_tstep.n_surf)]
        
        dyn_forces_excl_cs = [aero_tstep.dynamic_forces[i] * not_cs[i] for i in range(aero_tstep.n_surf)]
        dyn_force_diff = [aero_tstep.dynamic_forces[i] - dyn_forces_excl_cs[i] for i in range(aero_tstep.n_surf)]

        forces = mapping.aero2struct_force_mapping(
            aero_tstep.forces + aero_tstep.dynamic_forces,
            # forces_excl_cs + dyn_forces_excl_cs,
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
        
        header += ", y/s, cl, zdot, zddot"
        numb_col += 4
        lift_distribution = np.concatenate((lift_distribution, np.zeros((N_nodes, 4))), axis=1)
        
        N_elems = self.data.structure.num_elem
        theta_distribution = np.zeros((N_elems, 3))

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
                
                lift_distribution[inode, 6] = struct_tstep.pos_dot[inode, 2]  # zdot
                lift_distribution[inode, 7] = struct_tstep.pos_ddot[inode, 2]  # zddot
                
        for ielem in range(N_elems):
            theta_distribution[ielem, 0] = struct_tstep.psi[ielem, -1, 1]
            theta_distribution[ielem, 1] = struct_tstep.psi_dot[ielem, -1, 1]
            theta_distribution[ielem, 2] = struct_tstep.psi_ddot[ielem, -1, 1]
                
        # lift = np.sum(lift_distribution[:, 3])  # total lift force
        # u = np.linalg.norm(urel)
        # if u > 0.0:
        #     output = lift / (0.5 * self.settings['rho'] * (u**2) * total_area)
        # else: output = 0.0
        
        # cl = lift_distribution[:, 5]
        # clabs = np.abs(cl)
        # CL = np.mean(cl)
        # output = CL
        
        if mode == 'theta':
            output = np.mean(theta_distribution[:, 0])
        elif mode == 'theta_dot':
            output = np.mean(theta_distribution[:, 1])
        elif mode == 'theta_ddot':
            output = np.mean(theta_distribution[:, 2])
        elif mode == 'pos':
            output = np.mean(lift_distribution[:, 2])
        elif mode == 'pos_dot':
            output = np.mean(lift_distribution[:, 6])
        elif mode == 'pos_ddot':
            output = np.mean(lift_distribution[:, 7])
        elif mode == 'cl':
            output = np.mean(lift_distribution[:, 5])
        elif mode == 'tip_pos':
            output = struct_tstep.pos[N_nodes//2-1, 2]
        elif mode == 'tip_pos_dot':
            output = struct_tstep.pos_dot[N_nodes//2-1, 2]
        elif mode == 'tip_pos_ddot':
            output = struct_tstep.pos_ddot[N_nodes//2-1, 2]
        elif mode == 'tip_theta':
            output = struct_tstep.psi[N_elems//2-1, -1, 1]
        elif mode == 'tip_theta_dot':
            output = struct_tstep.psi_dot[N_elems//2-1, -1, 1]
        elif mode == 'tip_theta_ddot':
            output = struct_tstep.psi_ddot[N_elems//2-1, -1, 1]
        else:
            raise NotImplementedError(
                f"input_type {mode} is not yet implemented in extract_time_history()") 
        
        # output = np.sign(CL) * np.mean(clabs)
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
        if mode == 'tip_theta':
            controller = self.controller_implementation_tip_theta
            current_input = self.filter(current_input, 'lp', lp_freq0=0.1, lp_freq1=40)[0]
            # if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
            #     current_input = np.convolve(current_input, np.ones(self.settings['kernel_size']), 'same') / self.settings['kernel_size']
        elif mode == 'tip_theta_dot':
            controller = self.controller_implementation_tip_theta_dot
            current_input = self.filter(current_input, 'lp', lp_freq0=0.1, lp_freq1=40)[0]
            if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
                current_input = np.convolve(current_input, np.ones(self.settings['kernel_size']), 'same') / self.settings['kernel_size']
        elif mode == 'tip_theta_ddot':
            controller = self.controller_implementation_tip_theta_ddot
            # current_input = self.filter(current_input, 'lp', lp_freq0=0.1, lp_freq1=40)[0]
            if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
                current_input = np.convolve(current_input, np.ones(self.settings['kernel_size']), 'same') / self.settings['kernel_size']
        elif mode == 'tip_pos':
            controller = self.controller_implementation_tip_pos
            current_input = self.filter(current_input, 'lp', lp_freq0=0.1, lp_freq1=40)[0]
            # if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
            #     current_input = np.convolve(current_input, np.ones(self.settings['kernel_size']), 'same') / self.settings['kernel_size']
        elif mode == 'theta':
            controller = self.controller_implementation_theta
            current_input = self.filter(current_input, 'lp', lp_freq0=0.1, lp_freq1=40)[0]
            # if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
            #     current_input = np.convolve(current_input, np.ones(self.settings['kernel_size']), 'same')
        elif mode == 'theta_dot':
            controller = self.controller_implementation_theta_dot
            current_input = self.filter(current_input, 'lp', lp_freq0=0.1, lp_freq1=40)[0]
            if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
                current_input = np.convolve(current_input, np.ones(self.settings['kernel_size']), 'same') / self.settings['kernel_size']
        elif mode == 'theta_ddot':
            controller = self.controller_implementation_theta_ddot
            current_input = self.filter(current_input, 'lp', lp_freq0=0.1, lp_freq1=40)[0]
            if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
                current_input = np.convolve(current_input, np.ones(self.settings['kernel_size']), 'same') / self.settings['kernel_size']
        else:
            raise NotImplementedError('controller modes available are tip_theta, tip_pos, theta, theta_dot, and theta_ddot')
        
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
    
    
    def filter(self, data, mode=None, order=9, **kwargs):
        
        lp_freq0 = kwargs.get('lp_freq0', self.settings['cutoff_freq0'][0])
        lp_freq1 = kwargs.get('lp_freq1', self.settings['cutoff_freq0'][1])
        hp_freq0 = kwargs.get('hp_freq0', self.settings['cutoff_freq1'][0])
        hp_freq1 = kwargs.get('hp_freq1', self.settings['cutoff_freq1'][1])
        
        if self.settings['smoothing'] and self.settings['kernel_size'] > 1:
            data = np.convolve(data, np.ones(self.settings['kernel_size'])) / self.settings['kernel_size']
        
        if mode != 'fft':
            bl = sig.firwin(order, lp_freq1, pass_zero='lowpass', fs=(1/self.settings['dt']))
            bh = sig.firwin(order, hp_freq0, pass_zero='highpass', fs=(1/self.settings['dt']))
            try:
                data_lp = sig.filtfilt(bl, 1, data)
                data_hp = sig.filtfilt(bh, 1, data)
            except:
                data_lp = sig.lfilter(bl, 1, data)
                data_hp = sig.lfilter(bh, 1, data)

        else:
            data_fft = scipy.fft.fft(data, n=len(data))
            fft_freq = scipy.fft.fftfreq(data_fft.shape[0], d=self.settings['dt'])
            
            # print(fft_freq)
            
            data_lp = data_fft.copy()
            data_hp = data_fft.copy()

            
            
            data_lp[np.abs(fft_freq) <= lp_freq0] = 0
            data_lp[np.abs(fft_freq) > lp_freq1] = 0
            # print(data_lp)
            
            data_hp[np.abs(fft_freq) <= hp_freq0] = 0
            data_hp[np.abs(fft_freq) > hp_freq1] = 0
            # print(data_hp)
            
            data_lp = scipy.fft.ifft(data_lp, n=data_fft.shape[0]).real
            data_hp = scipy.fft.ifft(data_hp, n=data_fft.shape[0]).real
            
        return data_lp, data_hp

    def __exit__(self, *args):
        self.log.close()
