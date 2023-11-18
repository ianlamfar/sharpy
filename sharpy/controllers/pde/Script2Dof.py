import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define all wing parameters
L = 30
Vinf = 300

######## WING PARAMS ########
# chord = 5.8
# mass_wing = 50000
# thickness = 0.07 * chord
# E = 80  # in GPa
# xe = 0.05  # distance between CG and shear centre
# xa = 0.4  # between AC and shear centre, fraction of chord

# Ip = (1 / 16) * (mass_wing / L) * chord ** 2
# Iyy = (np.pi / 64) * chord * thickness ** 3
# J = np.pi * chord * thickness ** 3 / 16
# G = E / 2.5
# GJ = (10 ** 9) * G * J / L ** 2
# EI = (10 ** 9) * E * Iyy / L ** 4
######## WING PARAMS ########

# what's n_t and n_b
n_t = 3
n_b = 3

########## PAZY WING ###########
chord = 0.1
AR = 2 * 0.55 / 0.1
span = AR * chord
GJ = 7.2
EI = 3.11e3  # ei_inp, inplane bending
ea = 0.4475  # elastic axis/shear centre
xe = ea - 0.4510
xa = ea - 0.25
mass_wing = 5.5e-1 * span
Js = 3.03e-4
# Ip = (1 / 16) * (mass_wing / L) * chord ** 2  # might be Js?
Ip = Js
########## PAZY WING ###########



# Function to create the mass and stiffness matrices
# Function computes int_0^1 phi_1 phi_2 dz
def matrix_element(v1, v2):  # returns magnitude of modal coupling aij
    no_of_ele = 100
    dz = 0.01
    aij = 0.0
    for i in range(no_of_ele):  # finite integration
        z = i*dz + dz/2
        t1 = BasisFn(v1[0], v1[1], v1[2], z)
        t2 = BasisFn(v2[0], v2[1], v2[2], z)
        aij = aij + t1*t2*dz
    return aij


# Function to compute the basis functions for Galerkin's method
# Exact mode shapes of the unforced structural ODE are used as basis functions.
# Arguments: DoF (1 for twist); index > 0; derivative; coordinate
def BasisFn(arg1, k, d, z):
    if k == 0:
        if d == 0:
            eta = 0.78
            epsil = 0.02
            fn = (0.5/(1-eta + epsil))*(1 + np.tanh(10*(z-eta))) \
                + (0.5/(1-eta + epsil))*(1 - np.tanh(10*(z-1+epsil)))
        else:
            fn = 1
    else:
        if arg1 == 1:  # twist
            lk = (2*k-1)*np.pi/2
            repo_t = [np.sin(lk*z), np.cos(lk*z), -np.sin(lk*z)]
            fn = np.sqrt(2)*(lk**d)*repo_t[d]
        else:  # bending
            if k == 1:
                lk = 1.8745
            elif k == 2:
                lk = 4.6941
            else:
                lk = (2*k-1)*np.pi/2
            Bterm = (np.sinh(lk) - np.sin(lk))/(np.cos(lk) + np.cosh(lk))
            repo_b = [(np.cos(lk*z) - np.cosh(lk*z) - Bterm*(np.sin(lk*z)-np.sinh(lk*z))),
                      (-np.sin(lk*z) - np.sinh(lk*z) -
                       Bterm*(np.cos(lk*z) - np.cosh(lk*z))),
                      (-np.cos(lk*z) - np.cosh(lk*z) +
                       Bterm*(np.sin(lk*z) + np.sinh(lk*z))),
                      (np.sin(lk*z) - np.sinh(lk*z) +
                       Bterm*(np.cos(lk*z) + np.cosh(lk*z))),
                      (np.cos(lk*z) - np.cosh(lk*z) - Bterm*(np.sin(lk*z)-np.sinh(lk*z)))]
            fn = (lk**d)*repo_b[d]
    # print(arg1, k, d, z)
    return fn


M = np.zeros((n_t + n_b, n_t + n_b))
for i in range(n_t):
    for j in range(n_t):
        M[i, j] = Ip * matrix_element([1, i+1, 0], [1, j+1, 0])
for i in range(n_b):
    for j in range(n_b):
        M[n_t + i, n_t + j] = (mass_wing / L) * matrix_element([2, i+1, 0], [2, j+1, 0])
for i in range(n_t):
    for j in range(n_b):
        M[i, n_t + j] = -(mass_wing / L) * xe * chord * \
            matrix_element([1, i+1, 0], [2, j+1, 0])
        M[n_t + j, i] = M[i, n_t + j]
print('Mass matrix M:\n', M)

# Create the stiffness matrix (K) which can be used with K-V damping, sharpy equiv: elem_stiffness, FEM
K = np.zeros((n_t + n_b, n_t + n_b))
for i in range(n_t):
    for j in range(n_t):
        K[i, j] = -GJ * matrix_element([1, j+1, 2], [1, i+1, 0])
for i in range(n_b):
    for j in range(n_b):
        K[n_t + i, n_t + j] = EI * matrix_element([2, j+1, 4], [2, i+1, 0])
print('Stiffness matrix K:\n', K)

# Matrix for adding the effect of aileron deflection; assumes a 20% long aileron
Daint = np.zeros((n_t + n_b, 1))
for i in range(n_t):
    Daint[i] = xa * chord * matrix_element([1, 0, 0], [1, i+1, 0])
for i in range(n_b):
    Daint[n_t + i] = -matrix_element([1, 0, 0], [2, i+1, 0])
print('Aileron deflection Daint:\n', Daint)


# Matrix for evaluating the modal forcing function
Fmat = np.zeros((n_t + n_b, n_t))
for i in range(n_t):
    for j in range(n_t):
        Fmat[i, j] = matrix_element([1, i+1, 0], [1, j+1, 0])
for i in range(n_b):
    for j in range(n_t):
        Fmat[n_t + i, j] = matrix_element([2, i+1, 0], [1, j+1, 0])
print('Force matrix 1 Fmat:\n', Fmat)

Fmat2 = np.zeros((n_t + n_b, n_t))
for i in range(n_t):
    for j in range(n_b):
        Fmat2[i, j] = matrix_element([1, i+1, 0], [2, j+1, 0])
for i in range(n_b):
    for j in range(n_b):
        Fmat2[n_t + i, j] = matrix_element([2, i+1, 0], [2, j+1, 0])
print('Force matrix 2 Fmat2:\n', Fmat2)

# Matrix for calculating integral of theta for lift estimation
Lint = np.zeros((n_t, 1))
for i in range(n_t):
    Lint[i] = matrix_element([1, i+1, 0], [1, 0, 1])
print('Lift integral Lint:\n', Lint)

# Test the time constants
n = n_t + n_b
Cla = 2*np.pi/(1 + chord/L)
Minf = Vinf/340
f = 0.61*Vinf*Vinf*chord*Cla/np.sqrt(1 - Minf**2)
Q1 = f*(np.linalg.inv(M) @ np.block([[xa*chord*np.eye(n_t), np.zeros((n_t, n_b))], [np.zeros((n_b, n_t)), -np.eye(n_b)]]))
test_mat = np.block([[np.zeros((n, n)), np.eye(n)],
                     [-(np.linalg.inv(M) @ K) + Q1 @ np.block([[Fmat, np.zeros((n, n_b))]]), 0*-4*10**(-4)*(np.linalg.inv(M) @ K) + Q1 @ np.block([[np.zeros((n, n_t)), Fmat2]])]
                     ])
test_mat_short = np.block([[np.zeros((n_t, n_t)), np.eye(n_t)],
                           [-(np.linalg.inv(M[0:n_t, 0:n_t]) @ K[0:n_t, 0:n_t]), -2*10**(-5)*(np.linalg.inv(M[0:n_t, 0:n_t]) @ K[0:n_t, 0:n_t])]])
print('Time constants:')
print(np.linalg.eig(test_mat)[0])
print('Of which the twisting dynamics:')
print(np.linalg.eig(test_mat_short)[0])

# Set up the integration
par = [chord, L, xa, Vinf]
x0 = np.zeros((2*n+1, 1))  # end for lift error
for c1 in range(n_t):
    x0[c1] = 0.04/((c1+1)**2)*np.random.rand()
for c2 in range(n_b):
    x0[n_t+c2] = 0.04/((c2+1)**2)*np.random.rand()
print('x0:\n', x0)

def FT2dof(t, x):  # x: theta, xi, theta_dot, xi_dot
    n = n_t + n_b
    c, b, xa, V = par
    Cla = 2 * np.pi / (1 + c / b)
    Minf = V / 340
    f = 0.61 * V * V * c * Cla / np.sqrt(1 - Minf ** 2)

    eta = 4e-4  # Kelvin-Voigt damping factor, eta1=eta2=eta

    # da = 0.0  # a static deflection for now.
    ref_val = 0.1

    # With aileron feedback
    print(x[-1])
    da = -0.02 * x[-1] - 0.1 * (Lint.T @ x[:n_t] - ref_val)  # what's n_t and n_b?
    da = np.sign(da) * min(0.2, np.abs(da))

    # open loop
    # da = 0

    fda = 0.61 * V * V * c * 1.7 * da * Daint / np.sqrt(1 - Minf ** 2)

    dx = np.zeros((2 * n + 1, 1))

    xidot = x[n + n_t+1: 2 * n+1]  # rate of change of bending displacement
    Q1 = np.linalg.solve(M, np.block([[xa * c * np.eye(n_t), np.zeros((n_b, n_t))],
                                      [np.zeros((n_t, n_b)), -np.eye(n_b)]])
                         )
    # print(np.linalg.det(Q1))
    MKsolve = np.linalg.solve(M, K)  # constant a absorbed into it

    dx[:n] = np.expand_dims(x[n + 1: 2 * n + 1], 1)
    dx[n + 1: 2 * n + 1] = - MKsolve @ np.expand_dims(x[:n], 1) \
        - eta * MKsolve @ np.expand_dims(x[n + 1: 2 * n + 1], 1) \
            + f * Q1 @ (Fmat @ np.expand_dims(x[:n_t], 1) \
                + (1 / V) *np.expand_dims((Fmat2 @ xidot), 1)) + fda
    dx[-1] = Lint.T @ x[:n_t] - ref_val  # time integral of theta

    return dx.flatten()


def large_def(t, x):
    return x[0]


options = {'events': large_def}

t_span = [0, 10]
sol = solve_ivp(FT2dof, t_span, x0.flatten(), method='LSODA',
                dense_output=True, events=large_def)  # target: dx = 0
tsol = sol.t
xsol = sol.y.T
print('xsol:\n', xsol)


# Calculate wing deflection at the tip (or any other point for that matter)
def_hist = xsol[:, :n_t] @ Lint  # deflection time history
print(def_hist.shape, xsol.shape, xsol[:, :n_t+n_b].shape, n_t, n_b)
fig, ax = plt.subplots(2, dpi=200)
ax[0].plot(tsol, def_hist, linewidth=1)
ax[0].plot(tsol, 0.1 * np.ones_like(tsol), 'r', linewidth=1)
ax[0].grid(True)
ax[0].set_ylabel(r'$\int_0^L \theta(t,x) dx$')
# ax[0].set_xlabel('Time [s]')
ax[0].legend(['y(t)', 'r(t)'])
# ax[0].set_ylim((0, 0.2))

# Bending displacement
ax[1].plot(tsol, xsol[:, :n_t+n_b], linewidth=1)
ax[1].grid(True)
ax[1].set_ylabel('Mode amplitude')
ax[1].set_xlabel('Time [s]')
# ax[1].set_ylim((0, 20))
