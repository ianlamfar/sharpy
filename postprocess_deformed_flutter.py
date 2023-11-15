import numpy as np
import sys
sys.path.append('./pazy/sharpy-analysis-tools/')
from batch.sets import Actual
import linear.stability as stability
import os


def prepare_cases_db(sharpy_output_folder, source_cases, param_vec):
    cases_db = []

    for ith, src_path in enumerate(source_cases):
        res = Actual(sharpy_output_folder + src_path + '/*')
        res.case_parameter = param_vec[ith]
        res.systems = ['aeroelastic']
        res.load_bulk_cases('eigs', 'deflection', eigs_legacy=False)
        cases_db.append(res)

    return cases_db

def deflection_at_flutter(case, speed, deflection_list):
    case_im = case.aeroelastic.find_parameter_value(int(speed), return_idx=True)
    deflection_im = deflection_list[case_im][-1] / 0.55
    case_ip = case.aeroelastic.find_parameter_value(int(speed + 1), return_idx=True)
    deflection_ip = deflection_list[case_ip][-1] / 0.55
    d = np.interp(speed, [int(speed), int(speed + 1)], [deflection_im, deflection_ip])
    return d


# write eigenvalues:
def save_eigenvalues(case, output_folder):
    vel, eigs = case.eigs('aeroelastic')
    dest_file = output_folder + '/velocity_eigenvalues_alpha{:04g}.txt'.format(case.case_parameter * 100)
    np.savetxt(dest_file, np.column_stack((vel, eigs)))
    print('Saved velocity/eigs array to {}'.format(dest_file))

    v, damp, fn = stability.modes(vel, eigs, use_hz=True, wdmax=120*2*np.pi)
    np.savetxt(output_folder + 'velocity_damping_frequency_alpha{:04g}.txt'.format(case.case_parameter * 100),
               np.column_stack((v, damp, fn)))

    # filter out higher frequency modes for stability analysis only!
    v, damp, fn = stability.modes(vel, eigs, use_hz=True, wdmax=50*2*np.pi)
    vel, deflection = case.wing_tip_deflection(frame='g', alpha=case.case_parameter)
    np.savetxt(output_folder + '/velocity_wingtip_deflection_g_alpha{:04g}.txt'.format(case.case_parameter * 100),
               np.column_stack((vel, deflection)))

    flutter_speeds = stability.find_flutter_speed(v, damp, 0., vel_vmin=0.)
    print(flutter_speeds)
    deflection_flutter = []
    for ith, speed in enumerate(flutter_speeds):
        deflection_flutter.append(deflection_at_flutter(case, speed, deflection))
    np.savetxt(output_folder + '/flutter_speeds_alpha{:04g}.txt'.format(case.case_parameter * 100),
               np.column_stack((flutter_speeds, deflection_flutter)))
    return flutter_speeds


def main(M=16, N=1, Ms=20, alpha=[5], skin=False, te=False):
    M = M
    N = N
    Ms = Ms
    alpha = alpha
    for skin in [skin]:
        alpha_vec = np.array(alpha)
        sharpy_output_folder = './output/pazy/output/'
        output_folder = './output/pazy/output/flutter' # place to save results
        trailing_edge_weight = te
        legacy = False

        if skin:
            skin_str = 'skin_on'
        else:
            skin_str = 'skin_off'

        if not legacy:
            output_folder += 'sharpy_{}_te{:g}/'.format(skin_str, trailing_edge_weight)
        else:
            output_folder += 'sharpy_{}/'.format(skin_str)
        os.makedirs(output_folder, exist_ok=True)

        flutter_case_name = 'deformed_flutter_{}'.format(skin)

        if not legacy:
            source_cases = [f'pazy_M{M}N{N}Ms{Ms}_alpha{alpha*100:04g}_skin{skin:g}_te{trailing_edge_weight:g}/' for alpha in alpha_vec]
        else:
            source_cases = ['pazy_M{M}N{N}Ms{Ms}_alpha0025_skin{:g}_te0'.format(skin)] + [
                'pazy_M16N1Ms20_alpha{:04g}_skin{:g}'.format(
                    alpha * 100, skin) for alpha in alpha_vec[1:]]

        results = prepare_cases_db(sharpy_output_folder, source_cases, alpha_vec)
        for case in results:
            if len(case.cases['aeroelastic'].cases) == 0:
                continue
            flutter_speeds = save_eigenvalues(case, output_folder)
    return flutter_speeds


if __name__ == '__main__':
    main(M=8, N=2, Ms=10, alpha=[5], skin=False, te=False)
