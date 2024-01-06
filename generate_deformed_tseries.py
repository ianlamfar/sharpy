import numpy as np
import os
import sharpy.sharpy_main
import sys
sys.path.append('./pazy/pazy_model/')
from pazy_wing_model import PazyWing
import sharpy.utils.algebra as algebra


def generate_pazy_tseries(u_inf, case_name, output_folder='/output/', cases_subfolder='', **kwargs):
    # u_inf = 60
    alpha_deg = kwargs.get('alpha', 0.)
    rho = 1.225
    num_modes = 16
    gravity_on = kwargs.get('gravity_on', True)
    skin_on = kwargs.get('skin_on', False)
    trailing_edge_weight = kwargs.get('trailing_edge_weight', False)
    end_time = kwargs.get('end_time', 2)
    num_cores = kwargs.get('num_cores', 8)
    
    controller_id = kwargs.get('controller_id', {})
    controller_settings = kwargs.get('controller_settings', {})

    # Lattice Discretisation
    M = kwargs.get('M', 4)
    N = kwargs.get('N', 32)
    M_star_fact = kwargs.get('Ms', 10)

    # SHARPy nonlinear reference solution
    route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    case_route = route_test_dir + '/cases/' + cases_subfolder + '/' + case_name

    if not os.path.exists(case_route):
        os.makedirs(case_route)

    model_settings = kwargs.get('model_settings', {'skin_on': skin_on,
                                                    'discretisation_method': 'michigan',
                                                    'num_elem': N,
                                                    'surface_m': M,
                                                    'num_surfaces': 2})

    pazy = PazyWing(case_name=case_name, case_route=case_route, in_settings=model_settings)

    pazy.create_aeroelastic_model()
    if trailing_edge_weight:
        te_mass = 10e-3  # 10g
        trailing_edge_b = (pazy.get_ea_reference_line() - 1.0) * 0.1
        pazy.structure.add_lumped_mass((te_mass, pazy.structure.n_node//2, np.zeros((3, 3)),
                                        np.array([0, trailing_edge_b, 0])))
        pazy.structure.add_lumped_mass((te_mass, pazy.structure.n_node//2 + 1, np.zeros((3, 3)),
                                        np.array([0, trailing_edge_b, 0])))
    pazy.save_files()
    # print(M, N, M_star_fact, pazy.structure.n_elem)

    u_inf_direction = np.array([1., 0., 0.])
    dt = kwargs.get('dt', (pazy.aero.main_chord / M / u_inf))
    n_tsteps = int(end_time / dt)

    pazy.config['SHARPy'] = {
        'flow':
            ['BeamLoader',
             'AerogridLoader',
            #  'StaticTrim',
            #  'StaticCoupled',
            #  'AerogridPlot',
            #  'BeamPlot',
            #  'WriteVariablesTime',
             'DynamicCoupled',
            #  'Modal',
            #  'LinearAssembler',
            #  'AsymptoticStability',
             'SaveParametricCase',
             ],
        'case': pazy.case_name, 'route': pazy.case_route,
        'write_screen': 'on', 'write_log': 'on',
        'log_folder': output_folder + pazy.case_name + '/',
        'log_file': pazy.case_name + '.log'}

    pazy.config['BeamLoader'] = {
        'unsteady': True,
        'orientation': algebra.euler2quat([0., alpha_deg * np.pi / 180, 0])}

    # csgen_settings = {'dt': dt, 'deflection_file': ''}
    pazy.config['AerogridLoader'] = {
        'unsteady': True,
        'aligned_grid': 'on',
        'mstar': M_star_fact * M,
        'freestream_dir': u_inf_direction,
        'wake_shape_generator': 'StraightWake',
        'wake_shape_generator_input': {'u_inf': u_inf,
                                       'u_inf_direction': u_inf_direction,
                                       'dt': dt},
        # 'control_surface_deflection': model_settings['cs_deflection'],
        # 'control_surface_deflection_generator_settings': []
        }

    # pazy.config['StaticUvlm'] = {
    #     'rho': rho,
    #     'velocity_field_generator': 'SteadyVelocityField',
    #     'velocity_field_input': {
    #         'u_inf': u_inf,
    #         'u_inf_direction': u_inf_direction},
    #     # 'rollup_dt': dt,
    #     'print_info': 'on',
    #     # 'horseshoe': 'off',
    #     'num_cores': num_cores,
    #     # 'n_rollup': 0,
    #     # 'rollup_aic_refresh': 0,
    #     # 'rollup_tolerance': 1e-4
    #     }

    settings = dict()
    settings['StaticCoupled'] = {
        'print_info': True,
        'max_iter': 200,
        'n_load_steps': 4,  # default 4
        'tolerance': 1e-4,
        'relaxation_factor': 0.1,
        'aero_solver': 'StaticUvlm',
        'aero_solver_settings': {
            'rho': rho,
            'print_info': True,
            # 'horseshoe': 'off',
            'num_cores': num_cores,
            # 'n_rollup': 0,
            # 'rollup_dt': dt,
            # 'rollup_aic_refresh': 1,
            # 'rollup_tolerance': 1e-4,
            'vortex_radius': 1e-7,
            'velocity_field_generator': 'SteadyVelocityField',
            'velocity_field_input': {
                'u_inf': u_inf,
                'u_inf_direction': u_inf_direction}},
        'structural_solver': 'NonLinearStatic',
        'structural_solver_settings': {'print_info': True,
                                        'max_iterations': 200,
                                        'num_load_steps': 4,
                                        'delta_curved': 1e-6,
                                        'min_delta': 1e-8,
                                        'gravity_on': gravity_on,
                                        'gravity': 9.81}}
    
    pazy.config['StaticTrim'] = {'print_info': True,
                                 'solver': 'StaticCoupled',
                                 'solver_settings': settings['StaticCoupled'],
                                 'max_iter': 100,  # default 100
                                 'fz_tolerance': 1e-2,  # default 1e-2
                                 'fx_tolerance': 1e-2,  # default 1e-2
                                 'm_tolerance': 1e-2,  # default 1e-2
                                 'tail_cs_index': [0, 1],
                                 'thrust_nodes': [0],
                                 'has_cs': False,
                                 'has_thrust': False,
                                 'initial_alpha': np.radians(alpha_deg),
                                 'initial_deflection': 0.0,
                                 'initial_thrust': 0.0,
                                 'initial_angle_eps': np.radians(0.01),
                                 'initial_thrust_eps': 0.1,
                                 'relaxation_factor': 0.2,  # default 0.2
                                 'has_cs': False,
                                 'has_thrust': False,
                                 'save_info': True,
                                 }

    # pazy.config['AerogridPlot'] = {
    #                             #    'folder': route_test_dir + output_folder,
    #                              'include_rbm': 'off',
    #                              'include_applied_forces': 'on',
    #                              'minus_m_star': 0}

    # pazy.config['AeroForcesCalculator'] = {'folder': route_test_dir + '/{:s}/forces'.format(output_folder),
    #                                      'write_text_file': 'on',
    #                                      'text_file_name': pazy.case_name + '_aeroforces.csv',
    #                                      'screen_output': 'on',
    #                                      'unsteady': 'off'}

    # pazy.config['BeamPlot'] = {
    #                         #    'folder': route_test_dir + output_folder,
    #                          'include_rbm': 'off',
    #                          'include_applied_forces': 'on'}

    # pazy.config['BeamCsvOutput'] = {
    #                                 # 'folder': route_test_dir + output_folder,
    #                               'output_pos': 'on',
    #                               'output_psi': 'on',
    #                               'screen_output': 'on'}

    # pazy.config['WriteVariablesTime'] = {
    #                                     # 'folder': route_test_dir + output_folder,
    #                                     'structure_variables': ['pos'],
    #                                     'structure_nodes': list(range(0, pazy.structure.n_node//2))}

    pazy.config['SaveParametricCase'] = {
                                        #  'folder': route_test_dir + output_folder + pazy.case_name + '/',
                                         'save_case': 'off',
                                         'parameters': {'u_inf': u_inf}}

    settings = dict()
    settings['NonLinearDynamicPrescribedStep'] = {'print_info': 'on',
                                                  'max_iterations': 200,  # default 950
                                                  'delta_curved': 1e-1,  # default 1e-6
                                                  'min_delta': 1e-3,  # default 1e-8
                                                  'newmark_damp': 1e-2,  # default 5e-4
                                                  'relaxation_factor': 0, # default 0.3
                                                  'gravity_on': gravity_on,
                                                  'gravity': 9.81,
                                                  'num_steps': n_tsteps,
                                                  'num_load_steps': 4, # default 1
                                                  'dt': dt}
    
    settings['StepUvlm'] = {'print_info': 'on',
                            # 'horseshoe': 'off',
                            'num_cores': num_cores,
                            # 'n_rollup': 100,
                            'convection_scheme': 2,
                            # 'rollup_dt': dt,
                            # 'rollup_aic_refresh': 1,
                            # 'rollup_tolerance': 1e-4,
                            'velocity_field_generator': 'SteadyVelocityField',
                            'velocity_field_input': {'u_inf': u_inf,
                                                     'u_inf_direction': [1., 0., 0.]},
                            'rho': rho,
                            'n_time_steps': n_tsteps,
                            'dt': dt,
                            'gamma_dot_filtering': 3,
                           'vortex_radius': 1e-10}

    settings['DynamicCoupled'] = {'print_info': 'on',
                                  'structural_substeps': 0,  # default 10, major source of slowdown, 0 is fully coupled
                                  'dynamic_relaxation': 'on',
                                  'cleanup_previous_solution': 'on',
                                  'structural_solver': 'NonLinearDynamicPrescribedStep',
                                  'structural_solver_settings': settings['NonLinearDynamicPrescribedStep'],
                                  'aero_solver': 'StepUvlm',
                                  'aero_solver_settings': settings['StepUvlm'],
                                  'fsi_substeps': 200,
                                  'fsi_tolerance': 1e-2,  # default 1e-6
                                  'relaxation_factor': 0.0,  # default 0.2, relaxation causes slowdown but enhances stability
                                  'minimum_steps': 0,  # min steps before convergence
                                  'relaxation_steps': 0,  # default 150
                                  'final_relaxation_factor': 0.0,
                                  'n_time_steps': n_tsteps,
                                  'dt': dt,
                                  'include_unsteady_force_contribution': 'off',
                                  'controller_id': controller_id,
                                  'controller_settings': controller_settings,
                                  'postprocessors': [],
                                #   'postprocessors_settings': {'BeamLoads': {'folder': route_test_dir + output_folder,
                                #                                             'csv_output': 'off'},
                                #                               'BeamPlot': {'folder': route_test_dir + output_folder,
                                #                                            'include_rbm': 'on',
                                #                                            'include_applied_forces': 'on'},
                                #                               'StallCheck': {},
                                #                               'AerogridPlot': {
                                #                                   'u_inf': u_inf,
                                #                                   'folder': route_test_dir + output_folder,
                                #                                   'include_rbm': 'on',
                                #                                   'include_applied_forces': 'on',
                                #                                   'minus_m_star': 0},
                                #                             }
                                  }
    
    pazy.config['DynamicCoupled'] = settings['DynamicCoupled']
    pazy.config.write()
    out = sharpy.sharpy_main.main(['', pazy.case_route + '/' + pazy.case_name + '.sharpy'])
    return out


if __name__== '__main__':
    from datetime import datetime
    u_inf_vec = np.linspace(10, 90, 81)
    # u_inf_vec = [83]

    alpha = 5
    gravity_on = False
    skin_on = False
    trailing_edge_weight = False

    M = 4
    N = 1
    Ms = 10

    batch_log = 'batch_log_alpha{:04g}'.format(alpha*100)

    with open('./{:s}.txt'.format(batch_log), 'w') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('SHARPy launch - START\n')
        f.write("date and time = %s\n\n" % dt_string)

    for i, u_inf in enumerate(u_inf_vec):
        print('RUNNING SHARPY %f %f\n' % (alpha, u_inf))
        case_name = 'pazy_uinf{:04g}_alpha{:04g}'.format(u_inf*10, alpha*100)
        try:
            generate_pazy_tseries(u_inf, case_name,
                          output_folder='/output/pazy_M{:g}N{:g}Ms{:g}_alpha{:04g}_skin{:g}_te{:g}/'.format(
                              M, N, Ms, alpha*100, skin_on, trailing_edge_weight),
                          cases_subfolder='/M{:g}N{:g}Ms{:g}_skin{:g}_te{:g}/'.format(
                              M, N, Ms, skin_on, trailing_edge_weight),
                          M=M, N=N, Ms=Ms, alpha=alpha,
                          gravity_on=gravity_on,
                          skin_on=skin_on,
                          trailing_edge_weight=trailing_edge_weight)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('./{:s}.txt'.format(batch_log), 'a') as f:
                f.write('%s Ran case %i :::: u_inf = %f\n\n' % (dt_string, i, u_inf))
        except AssertionError:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            with open('./{:s}.txt'.format(batch_log), 'a') as f:
                f.write('%s ERROR RUNNING case %f\n\n' % (dt_string, u_inf))


