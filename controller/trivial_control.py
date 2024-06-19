# ========================
# ARC control via LMPC with error dynamics regression
# Authors: Luca Murra, Massimo Romano, Francesco Giarrusso
# Prof. Giuseppe Oriolo, Nicola Scianca
#
# This is the main code for the simulations of the LMPC technique applied to a kinematic bicycle model robot
# ========================


# ========================
# Section 1: Importing Libraries
# ========================
import numpy as np 
import casadi as cs
#import animation
import time
from collections import namedtuple
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Our Functions
from utils import track as trck, model, animation

EULER = 0
RUNGE_KUTTA = 1

# ========================
# Section 2: Main 
# ========================

def run(tr, lap, integration):

    # Parameters
    N = 5000       # Discrete time length
    steps = N       # Steps to complete trajectory (or N if not completed)
    delta_sim = 0.1 # Simulation step
    a_max = 12    # Maximum acceleration 
    a_min = -4*a_max    # Maximum brake
    da_max = a_max/0.1    # Maximum jerk allowed
    del_max = 0.6     # Maximum steering angle - related to maximum curvature allowed
    ddel_max = 15 # Maximum angular velocity of steering allowed
    v_max = 150/3.6 # Maximum velocity (m/s)

    # State and Input definitions
    x = np.zeros((8, N+1))      # x = [X, Y, phi, beta, v, s, n, xi]
    u = np.zeros((2, N))        # u = [a, delta]

    # Initial conditions
    x0 = np.concatenate((tr.start(), [0, 0, tr.s_low, 0, 0]))
    x[:, 0] = x0

    x_pred_record = []
    elapsed_time = np.zeros(N)

    safeSet = []

    f, p = model.get_car_kinematic_model(tr)

    overshoot_steps = 0

    # First lap => Trivial Control
    for j in range(N):

        s = x[5, j]     # Curvilinear abscissa
        xi = x[7, j]    # Error in orientation
        v = x[4, j]     # Velocity

        if s < tr.s_high:
            perc = x[5, j] / tr.s_high*100
            print("Step nr. %4d - [%s%s] %5.2f%s   " % (j, '█'*int(perc/2), '-'*int(50-perc/2), perc, '%'), end = '\r')
        else:
            if overshoot_steps == 0:
                print(f"Step nr. %4d - [%s] 100%s   " % (j, '█'*50, '%'))
            print("Overshooting%s" % ('.'*overshoot_steps), end = '\r')
            overshoot_steps += 1
            if overshoot_steps > 30:
                break

        if v < v_max:
            u[0, j] = a_max # Set the acceleration as a constant very small to control only steering angle
        else:
            u[0,j] = 0

        u[1, j] = np.arctan2(np.sin(-xi), np.cos(-xi)) #tr.theta(s) - phi

        if u[1, j] > del_max:
            print("Delta out of the upper bound!!!")
            u[1, j] = del_max
        elif u[1, j] < -del_max:
            print("Delta out of the lower bound!!!")
            u[1, j] = -del_max
        
        safeSet.append([
            x[4, j],    # v
            x[5, j],    # s
            x[6, j],    # n
            x[7, j],    # xi
            lap,        # lap
            j           # time
        ])

        # integrate
        x[3, j] = np.arctan(p.Lr / (p.Lr + p.Lf) * np.tan(u[1, j]))     # beta, in practice it's an input transformation

        if integration == EULER:
            dx = f(x[:,j], u[:, j]).full().squeeze()
        elif integration == RUNGE_KUTTA:
            # Define intermediate variables k1, k2, k3, and k4
            k1 = f(x[:, j], u[:, j]).full().squeeze()
            k2 = f(x[:, j] + 0.5 * delta_sim * k1, u[:, j]).full().squeeze()
            k3 = f(x[:, j] + 0.5 * delta_sim * k2, u[:, j]).full().squeeze()
            k4 = f(x[:, j] + delta_sim * k3, u[:, j]).full().squeeze()

            dx = (k1 + 2 * k2 + 2 * k3 + k4) /6

        x[:, j+1] = x[:, j] + delta_sim * dx    # Runge-Kutta 4th order integration

    cost = 0

    for i in range(len(safeSet)-1, -1, -1):
        cur_x = safeSet[i]      # [v, s, n, xi, lap, t]
        cost += 1 if cur_x[1] < tr.s_high else 0
        safeSet[i].append(cost) # [v, s, n, xi, lap, t, cost]

    print("\nIteration cost:", cost)
    #print("Elements in safeSet:", len(safeSet))
    #with open('costs.txt', 'a') as file:
    #    file.write(f"Iteration cost: {cost}\n")

    return safeSet