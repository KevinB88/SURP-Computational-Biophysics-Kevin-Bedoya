

import func_calc_mfpt_optimized as calc_op
import func_tab as tb
import func_plots as plt

import multiprocessing as mp
# import filepaths as fp
from datetime import datetime
from functools import partial
import time
import numpy as np


# pre-allocated containers for the pre-compiled/optimized code (using numba JIT decorators)
def initialize_layers(M, N):
    # Initialize layers outside of solve_mass_decay
    diffusive_layer = np.zeros((2, M, N), dtype=np.float64)
    advective_layer = np.zeros((2, M, N), dtype=np.float64)
    return diffusive_layer, advective_layer

def set_cores(core_input):
    fp.default_cpu_core_count_fp = 6
    if core_input > fp.default_cpu_core_count_fp:
        core_input = fp.default_cpu_core_count_fp
    return core_input


def solve(N_param, rg_param, ry_param, v_param, w_param):
    M = rg_param
    N = ry_param
    d_c = 1
    radius = 1
    v = v_param

    diffusive, advective = calc_op.initialize_layers(rg_param, ry_param)

    # mfpt = calc.solve_mass_decay(M, N, radius, d_c, sim_time, True, 1, w_param, w_param, v, N_param, True, ext_fact)
    # mfpt = calc.solve_mass_decay(M, N, radius, d_c, w_param, w_param, v, N_param)
    mfpt = calc_op.solve_mass_decay(M, N, radius, d_c, w_param, w_param, v, N_param, diffusive, advective)

    print(f'Microtubule Configuration: {N_param}')
    print()
    return {f'W: {w_param}', f'MFPT: {mfpt}'}


def parallel_process(rg_param, ry_param, v_param, N_list, w_low_bound, w_high_bound, cores):
    w_list = []
    lower_bound = w_low_bound
    upper_bound = w_high_bound
    for x in range(lower_bound, upper_bound+1):
        w_list.append(10 ** x)

    print(w_list)

    if len(w_list) < cores:
        process_count = len(v_list)
    else:
        process_count = cores

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_filepath = tb.create_directory(fp.mfpt_data_fp, current_time)

    for n in range(len(N_list)):
        with mp.Pool(processes=process_count) as pool:
            mfpt_results = pool.map(partial(solve, N_list[n], rg_param, ry_param, v_param), w_list)
        print(mfpt_results)

        tb.data_extraction_pandas(mfpt_results, data_filepath, f'MFPT_Results_N={len(N_list[n])}_v={v_param}_')
        time.sleep(1)

    plt.plot_all_csv_in_directory(data_filepath, N_list, data_filepath, f'MFPT_versus_W_v={v_param}_', True)


def solve_velocity(N_param, rg_param, ry_param, w_param, v_param):
    M = rg_param
    N = ry_param
    d_c = 1
    radius = 1
    v = v_param * -1

    diffusive, advective = calc_op.initialize_layers(rg_param, ry_param)

    # mfpt = calc.solve_mass_decay(M, N, radius, d_c, sim_time, True, 1, w_param, w_param, v, N_param, True, ext_fact)
    # mfpt = calc.solve_mass_decay(M, N, radius, d_c, w_param, w_param, v, N_param)
    mfpt = calc_op.solve_mass_decay(M, N, radius, d_c, w_param, w_param, v, N_param, diffusive, advective)

    print(f'Microtubule Configuration: {N_param}')
    print()
    return {f'W: {v_param}', f'MFPT: {mfpt}'}

def parallel_process_velocity(N_list, rg_param, ry_param, w_param, v_list, cores):

    print("velocity list: ", v_list)

    if len(v_list) < cores:
        process_count = len(v_list)
    else:
        process_count = cores

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_filepath = tb.create_directory(fp.mfpt_data_fp, current_time)

    for n in range(len(N_list)):
        with mp.Pool(processes=process_count) as pool:
            mfpt_results = pool.map(partial(solve_velocity, N_list[n], rg_param, ry_param, w_param), v_list)
        print(mfpt_results)

        tb.data_extraction_pandas(mfpt_results, data_filepath, f'MFPT_Results_N={len(N_list[n])}_W={w_param}_')
        time.sleep(1)

    plt.plot_all_csv_in_directory(data_filepath, N_list, data_filepath, f'MFPT_versus_v_W={w_param}_', True)


'''
    Parallel processing is leveraged to calculate multiple MFPT values simultaneously. 
    
    In ths example below, we calculate MFPT for different microtubule configurations and an increasing switch rate 
    (w), from 10^-2 to 10^4, for a fixed velocity v=10
    
'''
if __name__ == "__main__":

    rings = 32
    rays = 32
    # available cores, this parameter must be adjusted relative to the number of cores available on your computer
    a_cores = set_cores(6)
    v = 10

    microtubule_configs = [
        [0, 8, 16, 24],
        [0, 4, 8, 12, 16, 20, 24, 28],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    ]

    parallel_process(rings, rays, v, microtubule_configs, -2, 4, a_cores)






