from numba import njit
import numpy as np
import math

'''
    Optimizations are made using the Python numba library, leveraging JIT decorators to allow single-time compilation
    for an interpreted langauge.
'''

'''
    The only difference aside from the JIT decorators is the "initialize_layers" function, which initializes two
    static sized containers for the diffusive and advective layer. These containers are then used as input to the
    MFPT calculator function. 

'''
def initialize_layers(rings, rays):
    diffusive_layer = np.zeros((2, rings, rays), dtype=np.float64)
    advective_layer = np.zeros((2, rings, rays), dtype=np.float64)
    return diffusive_layer, advective_layer

@njit
def u_density(phi, k, m, n, d_radius, d_theta, d_time, central, rings, rho, dict_index, a, b, tube_placements):
    current_density = phi[k][m][n]
    component_a = ((m+2) * j_r_r(phi, k, m, n, d_radius, rings)) - ((m+1) * j_l_r(phi, k, m, n, d_radius, central))
    component_a *= d_time / ((m+1) * d_radius)
    component_b = (j_r_t(phi, k, m, n, d_radius, d_theta)) - (j_l_t(phi, k, m, n, d_radius, d_theta))
    component_b *= d_time / ((m+1) * d_radius * d_theta)

    if n == tube_placements[dict_index]:  # Check if 'n' matches the microtubule angle
        component_c = (a * phi[k][m][n]) * d_time - (((b * rho[k][m][n]) * d_time) / ((m+1) * d_radius * d_theta))
    else:
        component_c = 0

    return current_density - component_a - component_b - component_c

@njit
def u_tube(rho, phi, k, m, n, a, b, v, d_time, d_radius, d_theta):
    j_l = v * rho[k][m][n]
    j_r = 0 if m == len(phi[k][m]) - 1 else v * rho[k][m+1][n]
    return rho[k][m][n] - ((j_r - j_l) / d_radius) * d_time + (a * phi[k][m][n] * (m+1) * d_radius * d_theta) * d_time - b * rho[k][m][n] * d_time


@njit
def u_center(phi, k, d_radius, d_theta, d_time, curr_central, rho, tube_placements, v):
    # Calculate total_sum with an explicit loop instead of a generator expression
    total_sum = 0.0
    for n in range(len(phi[k][0])):
        total_sum += j_l_r(phi, k, 0, n, d_radius, curr_central)
    total_sum *= (d_theta * d_time) / (math.pi * d_radius)

    # Calculate diffusive_sum
    diffusive_sum = curr_central - total_sum

    # Calculate advective_sum with a loop
    advective_sum = 0.0
    for angle in tube_placements:
        j_l = rho[k][0][angle] * v
        advective_sum += (abs(j_l) * d_time) / (math.pi * d_radius * d_radius)

    # Return the combined sum
    return diffusive_sum + advective_sum

@njit
def calc_mass(phi, rho, k, d_radius, d_theta, curr_central, rings, rays, tube_placements):
    mass = 0.0
    for m in range(rings):
        for n in range(rays):
            mass += phi[k][m][n] * (m+1)
    mass *= (d_radius * d_radius) * d_theta

    microtubule_mass = 0.0
    for angle in tube_placements:
        for m in range(rings):
            microtubule_mass += rho[k][m][angle] * d_radius

    return (curr_central * math.pi * d_radius * d_radius) + mass + microtubule_mass


@njit
def calc_loss_mass_j_r_r(phi, k, d_radius, d_theta, rings, rays):
    total_sum = sum(j_r_r(phi, k, rings-2, n, d_radius, 0) for n in range(rays))
    total_sum *= rings * d_radius * d_theta
    return total_sum

@njit
def calc_loss_mass_derivative(mass_container, d_time):
    return np.diff(mass_container) / d_time

@njit
def j_r_r(phi, k, m, n, d_radius, rings):
    curr_ring = phi[k][m][n]
    next_ring = 0 if m == rings - 1 else phi[k][m+1][n]
    return -1 * ((next_ring - curr_ring) / d_radius)

@njit
def j_l_r(phi, k, m, n, d_radius, central):
    curr_ring = phi[k][m][n]
    prev_ring = central if m == 0 else phi[k][m-1][n]
    return -1 * ((curr_ring - prev_ring) / d_radius)

@njit
def j_r_t(phi, k, m, n, d_radius, d_theta):
    b = len(phi[k][m])
    return -1 * (phi[k][m][(n+1) % b] - phi[k][m][n]) / ((m+1) * d_radius * d_theta)

@njit
def j_l_t(phi, k, m, n, d_radius, d_theta):
    b = len(phi[k][m])
    return -1 * (phi[k][m][n] - phi[k][m][(n-1) % b]) / ((m+1) * d_radius * d_theta)

@njit
def solve_mass_decay(rings, rays, r, d, a, b, v, tube_placements, diffusive_layer, advective_layer):
    if len(tube_placements) > rays:
        raise IndexError(f'Too many microtubules requested: {len(tube_placements)}, within domain of {rays} angular rays.')

    for i in range(len(tube_placements)):
        if tube_placements[i] < 0 or tube_placements[i] > rays:
            raise IndexError(f'Angle {tube_placements[i]} is out of bounds, your range should be [0, {rays-1}]')

    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta * d_radius * d_radius)) / (2 * d)
    phi_center = 1 / (math.pi * (d_radius * d_radius))

    mass_retained = 0
    m_f_p_t = 0
    k = 0

    while k == 0 or mass_retained > 0.01:
        net_current_out = 0
        for m in range(rings):
            angle_index = 0
            for n in range(rays):
                if m == rings - 1:
                    diffusive_layer[1][m][n] = 0
                else:
                    diffusive_layer[1][m][n] = u_density(diffusive_layer, 0, m, n, d_radius, d_theta, d_time, phi_center, rings, advective_layer, angle_index, a, b, tube_placements)
                    if n == tube_placements[angle_index]:
                        advective_layer[1][m][n] = u_tube(advective_layer, diffusive_layer, 0, m, n, a, b, v, d_time, d_radius, d_theta)
                        if angle_index < len(tube_placements)-1:
                            angle_index += 1
                if m == rings - 2:
                    net_current_out += j_r_r(diffusive_layer, 0, m, n, d_radius, 0) * rings * d_radius * d_theta
        m_f_p_t += net_current_out * k * d_time * d_time
        k += 1

        mass_retained = calc_mass(diffusive_layer, advective_layer, 0, d_radius, d_theta, phi_center, rings, rays, tube_placements)
        phi_center = u_center(diffusive_layer, 0, d_radius, d_theta, d_time, phi_center, advective_layer, tube_placements, v)
        diffusive_layer[0] = diffusive_layer[1]
        advective_layer[0] = advective_layer[1]

    return m_f_p_t

