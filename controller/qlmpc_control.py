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
import time
from matplotlib import pyplot as plt

# Our Functions
from utils import model

EULER = 0
RUNGE_KUTTA = 1

#   safeSet = [[v, s, n, xi, t, cost]]
#                0 1  2  3   4    5

def areNear(state1, state2):
    threshold_s = 1e-3
    threshold_n = 1e-2
    threshold_xi = 1e-2

    y = np.abs(state1 - state2)

    return y[1] < threshold_s and y[2] < threshold_n and y[3] < threshold_xi

def isInSS(state, safeSet):
    # Returns True if the state is in the safeSet
    for ss in safeSet:
        if areNear(np.array(ss[:4]), np.array(state[:4])):
            return True
            
def isInCurrentSS(state, currentSafeSet):
    # Returns True if the state is in the current safeSet
    for stateInTable in currentSafeSet:
        if areNear(np.array(stateInTable[:4]), np.array(state[:4])):
            return True

def indexInSS(state, Q_table):   # state: [v, s, n, xi]

    idx = -1
    for i in range(len(Q_table)):
        q_value = Q_table[i]
        cft_v = cs.le(cs.fabs(state[0] - q_value[0]), 1e-2)
        cft_s = cs.le(cs.fabs(state[1] - q_value[1]), 1e-2)
        cft_n = cs.le(cs.fabs(state[2] - q_value[2]), 1e0)
        cft_xi = cs.le(cs.fabs(state[3] - q_value[3]), 1e-2)

        #all_conditions_met = cs.logic_and(cs.logic_and(cs.logic_and(cft_v, cft_s), cft_n), cft_xi)
        
        all_conditions_met = cs.logic_and(cs.logic_and(cft_s, cft_n), cft_xi)   # Ignore speed
        
        idx = cs.if_else(all_conditions_met, i, idx)
        
    return idx

def get_q_table_cost(index, Q_table):
    cost = 1e10
    
    for c in range(len(Q_table)):
        cost = cs.if_else(index == c, Q_table[c][-1], cost)
    
    return cost

# ========================
# Section 2: Main 
# ========================

def run (tr, safeSet, total_laps):

    # Optimizer Definition
    opti = cs.Opti() # Instance of Casadi Opti class (the Optimizer of Casadi)
    # Options of the Optimizer
    # 1. ipopt.print_level: Level of verbosity
    # 2. expand: if True, simplifies symbolic expressions before giving to the solver 
    p_opts = {"ipopt.print_level": 1, "expand": True, 'print_time': False} # Principal options
    s_opts = {} #{"max_iter": 1} # Secondary Options
    opti.solver("ipopt", p_opts, s_opts) # Definition of the solver ipopt (solver for nonlinear objective and constraints)

    # Parameters
    N_lmpc = 3             # LMPC horizon
    num_overshoot = 2*N_lmpc  # Nr. of overshooting steps
    laps = total_laps       # Nr. of laps
    N = 10000                # Maximum number of steps per lap
    steps = N               # Steps to complete trajectory (or N if not completed)
    delta = 0.1             # Time step
    a_max = 12              # Maximum acceleration 
    a_min = -4*a_max        # Maximum brake
    del_max = 0.6           # Maximum steering angle - related to maximum curvature allowed
    ddel_max = 0.05         # Maximum angular velocity of steering allowed
    v_max = 350/3.6         # Maximum velocity (m/s)
    EMERGENCY_LIMIT = 10    # Maximum number of consecutive emergency controls
    delta_weight = 0.001    # Cost weight of the delta
    crc = 0.01              # Control rate cost (best for now is 0.01)
    plotLap = True         # Choose to display plots at the end of each lap
    
    '''# MODEL DEBUG
    #f, p = model.get_reduced_linearized_model(tr, delta_lmpc)
    f, p = model.get_reduced_model(tr)

    state = np.array([0, 0, 0, 0], dtype=np.float64)
    state_norm = np.array([0, 0, 0, 0], dtype=np.float64)
    x = []
    y = []
    x_norm = []
    y_norm = []

    for i in range(100):
        u = [1, 0.01*i]

        A, B, C = model.get_jacobians_rk_here(tr, delta_lmpc, state, u)
        dx = (A @ state + B @ u + C - state).full().squeeze()
        state += dx
        x_y = tr.sn2xy(state[1], state[2])
        x.append(x_y[0])
        y.append(x_y[1])

        #dx_norm = delta_lmpc * f_norm(state_norm, u).full().squeeze()
        k1 = f(state_norm, u).full().squeeze()
        k2 = f(state_norm + 0.5 * delta_lmpc * k1, u).full().squeeze()
        k3 = f(state_norm + 0.5 * delta_lmpc * k2, u).full().squeeze()
        k4 = f(state_norm + delta_lmpc * k3, u).full().squeeze()

        dx_norm = (k1 + 2 * k2 + 2 * k3 + k4) *delta_lmpc/6    # Runge-Kutta 4th order integration
        state_norm += dx_norm
        x_y = tr.sn2xy(state_norm[1], state_norm[2])
        x_norm.append(x_y[0])
        y_norm.append(x_y[1])

    plt.plot(x, y, 'b', x_norm, y_norm, 'r')
    plt.legend(['Linearized', 'Normal'])
    plt.show()
    exit()'''
    
    # set up optimization problem
    start_time = time.time()
    X = opti.variable(4, N_lmpc+1)     # v, s, n, xi
    U = opti.variable(2, N_lmpc)

    x0_param = opti.parameter(4)
    opti.subject_to( X[:, 0] == x0_param )

    f, p = model.get_reduced_model(tr)

    for i in range(N_lmpc):  # Bounds on input derivatives
        opti.subject_to( X[:, i+1] == X[:, i] + delta * f(X[:, i], U[:, i]) )

        if i > 0:
            opti.subject_to( opti.bounded(-ddel_max, U[1, i] - U[1, i-1], ddel_max))

    # Constraints (8e)
    opti.subject_to( opti.bounded( -2*tr.w/5, X[2, :], 2*tr.w/5 ) )       # Stay within the bounds
    opti.subject_to( opti.bounded( 0, X[0, :], v_max) )             # Don't go faster than v_max

    opti.subject_to( opti.bounded(a_min, U[0, :], a_max) )          # Bounds on acceleration
    opti.subject_to( opti.bounded(-del_max, U[1, :], del_max) )     # Bounds on steering angle

    #print(opti)

    # State and Input definitions
    x = np.zeros((8, N+1))      # x = [X, Y, phi, beta, v, s, n, xi]
    u = np.zeros((2, N))        # u = [a, delta]

    # Initial conditions
    x0 = np.concatenate((tr.start(), [0, 0, tr.s_low, 0, 0]))
    x[:, 0] = x0

    x_pred_record = []
    elapsed_time = np.zeros(N)
    iter_costs = []
    samples_in_ss = []
    overshoot_steps = 0

    x_cars = []
    steps_cars = []

    f, p = model.get_car_kinematic_model(tr)

    # Just for debug
    def callback(i):
        if i > 10:
            plt.plot(opti.debug.value(X)[0, :], opti.debug.value(X)[1, :], label= str(i))

    for lap in range(laps):
        print("\nLMPC iteration nr.", lap)
        
        x_tmp = x.copy()
        u_tmp = u.copy()

        samples = len(safeSet)
        print("Points in use:", samples)

        samples_in_ss.append(samples)
        current_safeSet = []
        overshoot_steps = 0

        cost = 0
        index_ss = indexInSS(X[:, -1], safeSet)
        cost += get_q_table_cost(index_ss, safeSet)
        #cost += -cs.sumsqr(X[1,:])

        for t in range(N):
            if x_tmp[5, t] > tr.s_high:     # Overshooting the trajectory for further iterations
                #print("\nReached end of trajectory at iteration nr.", j)
                if x_tmp[4, t] < v_max:
                    u_tmp[0, t] = a_max
                else:
                    u_tmp[0, t] = 0
                u_tmp[1, t] = 0

                if overshoot_steps == 0:
                    print(f"Step nr. %4d - [%s] 100%s   " % (t, '█'*50, '%'))
                print("Overshooting%s" % ('.'*overshoot_steps), end = '\r')
                overshoot_steps += 1
                if overshoot_steps > num_overshoot:    # Further points than goal are needed for further iterations
                    steps = t
                    break
            else:
                if t > 0:
                    perc = x_tmp[5, t] / tr.s_high*100
                    print(f"Step nr. %4d - [%s%s] %5.2f%s   " % (t, '█'*int(perc/2), '-'*int(50-perc/2), perc, '%'), end = '\r')

                opti.set_value(x0_param, x_tmp[4:, t])      # Constraint (8b)

                if t > 0:
                    # set initial guess for next iteration
                    opti.set_initial(X, cs.horzcat(x_tmp[4:, t], x_pred[:, 2:], x_pred[:, -1]))
                    opti.set_initial(U, cs.horzcat(u_pred[:, 1:], u_pred[:, -1]))
                    #print(x_pred[:, 0])

                #cost += delta_weight * cs.sumsqr(U[1, :])  # Minimize delta

                for i in range(N_lmpc):
                    cost += cs.if_else(cs.le(X[1, i], 1), 1, 0)

                    '''if i == 0 and t > 0:
                        cost += crc * cs.sumsqr(U[:, i] - u_tmp[:, t])
                    else:
                        cost += crc * cs.sumsqr(U[:, i] - U[:, i-1])'''

                opti.minimize(cost)
                
                # solve NLP
                try:
                    sol = opti.solve()

                    u_tmp[:, t] = sol.value(U[:, 0])

                    u_pred = sol.value(U)
                    x_pred = sol.value(X)
                    x_pred_record.append(x_pred)

                    emergency_control = 0

                except Exception as e:
                    #print("\nEMERGENCY CONTROL!")
                    emergency_control +=1

                    if emergency_control >= EMERGENCY_LIMIT:
                        print(f"\nSTOP: lap {lap -1} is the best!")
                        break

                    if x_tmp[4, t] > 0:
                        u_tmp[0, t] = a_min/10
                    else:
                        u_tmp[0, t] = a_max /10

                    u_tmp[1, t] = -del_max * np.sign(x_tmp[6, t])
                
                    '''print(x[5, j])
                    x_err = cur_opti.debug.value(X)
                    cur_opti.debug.show_infeasibilities()
                    print("Can't go further at iteration nr.", j)
                    print("Predicted trajectory:", x_err)
                    print("Actual curvature:", tr.k(x_err[1, 0]))
                    print("Furthest predicted curvature:", tr.k(x_err[1, -1]))
                    print(e)
                    steps = j-1
                    break'''
                    #exit()

            current_safeSet.append([
                x_tmp[4, t],    # v
                x_tmp[5, t],    # s
                x_tmp[6, t],    # n
                x_tmp[7, t],    # xi
            ])

            # integrate
            x_tmp[3, t] = np.arctan(p.Lr / (p.Lr + p.Lf) * np.tan(u[1, t]))     # beta, in practice it's an input transformation

            # Define intermediate variables k1, k2, k3, and k4
            k1 = f(x_tmp[:, t], u_tmp[:, t]).full().squeeze()
            k2 = f(x_tmp[:, t] + 0.5 * delta * k1, u_tmp[:, t]).full().squeeze()
            k3 = f(x_tmp[:, t] + 0.5 * delta * k2, u_tmp[:, t]).full().squeeze()
            k4 = f(x_tmp[:, t] + delta * k3, u_tmp[:, t]).full().squeeze()

            dx = (k1 + 2 * k2 + 2 * k3 + k4) /6    # Runge-Kutta 4th order integration

            x_tmp[:, t+1] = x_tmp[:, t] + delta * dx    

            if lap == -1 and t >= 0:    # Put the right lap (remember that it starts from backSight) and time step to debug the run

                #print(D_x, D_y)
                cur_pos = tr.sn2xy(x_tmp[5, t], x_tmp[6, t])   

                print('Current state:', x_tmp[4:, t])
                #print('Target:', target)
                #print('Final state:', x_pred[:, -1])
                print(u_pred[0, :])
                
                pred_x = []
                pred_y = []
                for k in range(x_pred.shape[1]):
                    ss = x_pred[:, k]

                    pivot = tr.sn2xy(ss[1], ss[2])
                    pred_x.append(pivot[0])
                    pred_y.append(pivot[1])

                tr.plot_track(plt)
                plt.plot(pred_x, pred_y, color = 'b', marker = '.')
                plt.scatter(cur_pos[0], cur_pos[1], color = 'r')
                plt.title('Lap ' + str(lap) + ', t = ' + str(t))

                plt.axis((min(min(pred_x), cur_pos[0]) - 1, max(max(pred_x), cur_pos[0]) + 1, min(min(pred_y), cur_pos[1]) - 1, max(max(pred_y), cur_pos[1]) + 1))
                plt.show()
            
            elapsed_time[t] = time.time() - start_time

        last_cost = 0 if emergency_control < EMERGENCY_LIMIT else 100

        for i in range(len(current_safeSet)-1, -1, -1):
            cur_x = current_safeSet[i]              # [v, s, n, xi]
            last_cost += 1 if cur_x[1] < tr.s_high else 0
            current_safeSet[i].append(last_cost)    # [v, s, n, xi, cost]

        if plotLap:
            plt.figure()

            tr.plot_track(plt)

            # Plot not touched elements of the safe set in black
            # Plot touched elements of the safe set in red
            x_old = []
            y_old = []
            x_common = []
            y_common = []
            for state in safeSet:
                if state[1] > tr.s_high:
                    break

                if isInCurrentSS(state, current_safeSet):
                    point = tr.sn2xy(state[1], state[2])
                    x_common.append(point[0])
                    y_common.append(point[1])
                else:
                    point = tr.sn2xy(state[1], state[2])
                    x_old.append(point[0])
                    y_old.append(point[1])

            plt.scatter(x_old, y_old, color='k')
            plt.scatter(x_common, y_common, color='r')

            # Plot new elements of the safe set in green
            x_new = []
            y_new = []
            v_new = []
            for state in current_safeSet:
                if state[1] > tr.s_high:
                        break
                
                point = tr.sn2xy(state[1], state[2])
                x_new.append(point[0])
                y_new.append(point[1])
                v_new.append(state[0] * 3.6)
            plt.scatter(x_new, y_new, c = v_new)
            cbar = plt.colorbar()
            cbar.set_label('Speed [km/h]')

            plt.title('Lap: '+ str(lap +1) + ', Iteration cost: ' + str(last_cost))
            plt.axis('equal')
            plt.show()

        if emergency_control >= EMERGENCY_LIMIT:
            break

        x = x_tmp.copy()
        u = u_tmp.copy()

        x_cars.append(x)
        steps_cars.append(steps-num_overshoot)

        for ss in current_safeSet:
            if not isInSS(ss, safeSet):
                safeSet.append(ss)

        iter_costs.append(last_cost)
        print("\nIteration cost:", last_cost)

        #with open('costs.txt', 'a') as file:
        #    file.write(f"Iteration cost: {cost}\n")

    plt_cost = plt.subplot(1,2,1)
    plt_ss = plt.subplot(1,2,2)
    plt_cost.plot(iter_costs)
    plt_cost.set_title('Iteration costs')
    plt_ss.plot(samples_in_ss)
    plt_ss.set_title('Samples in safe set')
    plt.show()
   
    #animation.animate_car(tr, x, u, x_pred_record, steps - num_overshoot, 10)
    
    return x_cars, steps_cars