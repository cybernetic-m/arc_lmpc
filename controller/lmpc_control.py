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
from utils import model, animation

EULER = 0
RUNGE_KUTTA = 1

#   safeSet = [[v, s, n, xi, lap, t, cost]]
#                0 1  2  3   4    5    6

def areNear(state1, state2):
    threshold_s = 1e-3
    threshold_n = 1e-2
    threshold_xi = 1e-2

    y = np.abs(state1 - state2)

    return y[1] < threshold_s and y[2] < threshold_n and y[3] < threshold_xi

def isInSS(state, safeSet, backSight):
    # Returns True if the state is in the safeSet
    for statesOfLap in safeSet[-backSight:]:
        for stateInTable in statesOfLap:
            if areNear(np.array(stateInTable[:4]), np.array(state[:4])):
                return True
            
def isInCurrentSS(state, currentSafeSet):
    # Returns True if the state is in the current safeSet
    for stateInTable in currentSafeSet:
        if areNear(np.array(stateInTable[:4]), np.array(state[:4])):
            return True

def get_D_S_J(state, K, safeSet, backSight):
    W = np.diag([0, 1, 0, 0])

    for statesOfLap in safeSet[-backSight:]:
        d_list = []

        for s in statesOfLap:
            if type(state) is list:
                npState = np.array(state)
            else:
                npState = state
            
            y = np.array(s[:4]) - npState
            d_norm = y.T @ W.T @ W @ y

            d_list.append(d_norm.__float__())

        indices = np.argsort(d_list).squeeze()
        nearestK = np.array(statesOfLap)[indices[:K]]

        for i in range(K):
            if indices[i] >= len(statesOfLap)-1:
                indices[i] = len(statesOfLap) -2

        nextK = np.array(statesOfLap)[indices[:K] +1]

        if not 'J' in locals():
            D = nearestK[0][:4].reshape(4, 1)
            S = nextK[0][:4].reshape(4, 1)
            J = nearestK[0][6]
        else:
            D = np.hstack((D, nearestK[0][:4].reshape(4, 1)))
            S = np.hstack((S, nextK[0][:4].reshape(4, 1)))
            J = np.hstack((J, nearestK[0][6]))

        for index in range(1, len(nearestK)):
            x = nearestK[index]
            nextX = nextK[index]

            D = np.hstack((D, x[:4].reshape(4, 1)))
            S = np.hstack((S, nextX[:4].reshape(4, 1)))
            J = np.hstack((J, x[6]))

    J = J.reshape(1, -1)
    return D, S, J

def getFromSafeSet(lap, time, safeSet):
    statesOfLap = safeSet[lap]

    for ss in statesOfLap:
        if ss[5] == time:
            return ss[:4]

# ========================
# Section 2: Main 
# ========================

def run (tr, safeSet, backSight, integration, total_laps):

    # Optimizer Definition
    opti = cs.Opti() # Instance of Casadi Opti class (the Optimizer of Casadi)
    # Options of the Optimizer
    # 1. ipopt.print_level: Level of verbosity
    # 2. expand: if True, simplifies symbolic expressions before giving to the solver 
    p_opts = {"ipopt.print_level": 1, "expand": True, 'print_time': False} # Principal options
    s_opts = {} #{"max_iter": 1} # Secondary Options
    opti.solver("ipopt", p_opts, s_opts) # Definition of the solver ipopt (solver for nonlinear objective and constraints)

    # Parameters
    K = 10                 # Nr. of nearest neighbors
    N_lmpc = 4              # LMPC horizon
    num_overshoot = 2*(K + N_lmpc)  # Nr. of overshooting steps
    laps = total_laps       # Nr. of laps
    N = 5000                # Maximum number of steps per lap
    steps = N               # Steps to complete trajectory (or N if not completed)
    delta = 0.1             # Time step
    a_max = 12              # Maximum acceleration 
    a_min = -4*a_max        # Maximum brake
    del_max = 0.6           # Maximum steering angle - related to maximum curvature allowed
    ddel_max = 0.05         # Maximum angular velocity of steering allowed
    v_max = 350/3.6         # Maximum velocity (m/s)
    EMERGENCY_LIMIT = 10    # Maximum number of consecutive emergency controls
    delta_weight = 1        # Cost weight of the delta
    crc = 0.01              # Control rate cost (best for now is 0.01)
    plotLap = False         # Choose to display plots at the end of each lap
    
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
    Lambda = opti.variable(K * backSight, 1)

    x0_param = opti.parameter(4)
    opti.subject_to( X[:, 0] == x0_param )

    for i in range(1, N_lmpc):  # Bounds on input derivatives
        opti.subject_to( opti.bounded(-ddel_max, U[1, i] - U[1, i-1], ddel_max))

    # Constraints (8e)
    opti.subject_to( opti.bounded( -2*tr.w/5, X[2, :], 2*tr.w/5 ) )  # Stay within the bounds
    opti.subject_to( opti.bounded( 0, X[0, :], v_max) )              # Don't go faster than v_max

    opti.subject_to( opti.bounded(a_min, U[0, :], a_max) )          # Bounds on acceleration
    opti.subject_to( opti.bounded(-del_max, U[1, :], del_max) )     # Bounds on steering angle

    # Constraints (8c)
    opti.subject_to( Lambda[:] >= 0 )
    opti.subject_to( cs.sum1(Lambda[:]) == 1 )

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

    for lap in range(backSight, backSight + laps):
        print("\nLMPC iteration nr.", lap - backSight)
        
        x_tmp = x.copy()
        u_tmp = u.copy()
        x_pred_record = []

        samples = 0
        for statesOfLap in safeSet[-backSight:]:
            samples += len(statesOfLap)
        print("Points in use:", samples)

        samples_in_ss.append(samples)
        current_safeSet = []
        overshoot_steps = 0

        lap_start_time = time.time()

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
                    z = S @ prev_lambda[:]      # z_j^lap

                    # set initial guess for next iteration
                    #opti.set_initial(X, cs.horzcat(x[4:, t], x_pred[:, 2:], z))
                    #opti.set_initial(U, cs.horzcat(u_pred[:, 1:], u_pred[:, -1]))
                    #print(x_pred[:, 0])
                else:
                    z = getFromSafeSet(lap-1, N_lmpc, safeSet)

                D, S, J = get_D_S_J(z, K, safeSet, backSight)

                cur_opti = opti.copy()

                cur_opti.subject_to( (cs.DM(D) @ Lambda)[1:] == X[1:, -1] )    # Constraint (8c)

                # Equation (8a)
                cost = 0
                cost += delta_weight * cs.sumsqr(U[1, :])  # Minimize delta

                for i in range(N_lmpc):
                    if t == 0:      # At the first iteration we don't have x_pred and u_pred
                        lin_state = safeSet[-1][i][:4]
                        lin_control = [a_max, 0]
                    else:   
                        lin_state = x_pred[:, i+1]
                        lin_control = u_pred[:, min(i+1, N_lmpc-1)]
                    
                    if integration == EULER:
                        A, B, C = model.get_jacobians_here(tr, delta, lin_state, lin_control)
                    elif integration == RUNGE_KUTTA:
                        A, B, C = model.get_jacobians_rk_here(tr, delta, lin_state, lin_control)

                    cur_opti.subject_to( X[:, i+1] == A @ X[:, i] + B @ U[:, i] + C )   # Constraint (8d)

                    cost += cs.if_else(cs.le(X[1, i], 1), 1, 0)

                    if i == 0 and t > 0:
                        cost += crc * cs.sumsqr(U[:, i] - u_tmp[:, t])
                    else:
                        cost += 100 * crc * cs.sumsqr(U[:, i] - U[:, i-1])

                cost += cs.DM(J) @ Lambda[:]

                cur_opti.minimize(cost)
                
                # solve NLP
                try:
                    sol = cur_opti.solve()

                    u_tmp[:, t] = sol.value(U[:, 0])

                    u_pred = sol.value(U)
                    x_pred = sol.value(X)
                    x_pred_record.append(x_pred)

                    prev_lambda = sol.value(Lambda)

                    #print(np.round(prev_lambda, 2))

                    #print("Chosen cost:", J @ prev_lambda)

                    emergency_control = 0

                except Exception as e:
                    #print("\nEMERGENCY CONTROL!")
                    emergency_control +=1

                    if emergency_control >= EMERGENCY_LIMIT:
                        print(f"\nSTOP: lap {lap - backSight -1} is the best!")
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
                lap,        # lap
                t           # time
            ])

            # integrate
            x_tmp[3, t] = np.arctan(p.Lr / (p.Lr + p.Lf) * np.tan(u[1, t]))     # beta, in practice it's an input transformation

            if integration == EULER:
                dx = f(x_tmp[:, t], u_tmp[:,  t]).full().squeeze()
            elif integration == RUNGE_KUTTA:
                # Define intermediate variables k1, k2, k3, and k4
                k1 = f(x_tmp[:, t], u_tmp[:, t]).full().squeeze()
                k2 = f(x_tmp[:, t] + 0.5 * delta * k1, u_tmp[:, t]).full().squeeze()
                k3 = f(x_tmp[:, t] + 0.5 * delta * k2, u_tmp[:, t]).full().squeeze()
                k4 = f(x_tmp[:, t] + delta * k3, u_tmp[:, t]).full().squeeze()

                dx = (k1 + 2 * k2 + 2 * k3 + k4) /6    # Runge-Kutta 4th order integration

            x_tmp[:, t+1] = x_tmp[:, t] + delta * dx    

            if lap == -1 and t >= 0:    # Put the right lap (remember that it starts from backSight) and time step to debug the run
                
                z_pos = tr.sn2xy(z[1], z[2])
                pred_x = []
                pred_y = []
                for k in range(x_pred.shape[1]):
                    ss = x_pred[:, k]

                    pivot = tr.sn2xy(ss[1], ss[2])
                    pred_x.append(pivot[0])
                    pred_y.append(pivot[1])
                
                #print(pred_x, pred_y)

                D_x = []
                D_y = []
                for k in range(D.shape[1]):
                    ss = D[:, k]

                    pivot = tr.sn2xy(ss[1], ss[2])
                    D_x.append(pivot[0])
                    D_y.append(pivot[1])

                #print(D_x, D_y)
                target = D @ prev_lambda
                target_pos = tr.sn2xy(target[1], target[2])

                cur_pos = tr.sn2xy(x_tmp[5, t], x_tmp[6, t])   

                #print('Current state:', x[4:, t])
                #print('Target:', target)
                #print('Final state:', x_pred[:, -1])
                print(u_pred[1, :])

                
                #tr.plot_track(plt)
                plt.scatter(D_x, D_y, color = 'k', marker = ',')   
                plt.plot(pred_x, pred_y, color = 'b', marker = '.')
                plt.scatter(cur_pos[0], cur_pos[1], color = 'r')                      
                plt.scatter(z_pos[0], z_pos[1], color = 'g')
                plt.scatter(target_pos[0], target_pos[1], color = 'y')
                plt.title('Lap ' + str(lap) + ', t = ' + str(t))
                

                #plt.axis((min(min(D_x), cur_pos[0]) - 1, max(max(D_x), cur_pos[0]) + 1, min(min(D_y), cur_pos[1]) - 1, max(max(D_y), cur_pos[1]) + 1))

                '''for i in range(100):
                    lam = np.random.rand(K*backSight, 1)

                    lam /= np.sum(lam)

                    if i == 0:
                        lam[:] = 0
                        lam[-1] = 1
                    elif i == 1:
                        lam[:] = 0
                        lam[-2] = 1

                    point_sn = (D @ lam).squeeze()
                    point = tr.sn2xy(point_sn[1], point_sn[2])

                    if i == 0:
                        plt.scatter(point[0], point[1], color = 'orange', marker = 'v')
                    elif i == 1:
                        plt.scatter(point[0], point[1], color = 'orange', marker = 'v')
                    else:
                        plt.scatter(point[0], point[1], color = 'b', marker = 'v')'''
                plt.show()
            
            elapsed_time[t] = time.time() - start_time

        last_cost = 0 if emergency_control < EMERGENCY_LIMIT else 100

        print("\nComputation time:", time.time() - lap_start_time)

        for i in range(len(current_safeSet)-1, -1, -1):
            cur_x = current_safeSet[i]              # [v, s, n, xi, lap, time]
            last_cost += 1 if cur_x[1] < tr.s_high else 0
            current_safeSet[i].append(last_cost)    # [v, s, n, xi, lap, time, cost]

        if plotLap:
            plt.figure(figsize=(10, 6))

            tr.plot_track(plt)

            # Plot not touched elements of the safe set in black
            # Plot touched elements of the safe set in red
            x_old = []
            y_old = []
            x_common = []
            y_common = []
            for statesOfLap in safeSet[-backSight:]:
                for state in statesOfLap:
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

            #plt.scatter(x_old[0:len(x_old):20], y_old[0:len(y_old):20], color='k', s = 20)
            #plt.scatter(x_common, y_common, color='r')

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

            plt.scatter(x_new[0:len(x_new):1], y_new[0:len(y_new):1], c = v_new[0:len(v_new):1], s = 30, cmap = 'autumn_r', zorder=10, vmin=0, vmax=v_max*3.6)
            
            cbar = plt.colorbar()
            cbar.set_label('Speed [km/h]')

            plt.title('Lap: '+ str(lap - backSight +1) + ', Iteration cost: ' + str(last_cost))
            plt.axis('equal')
            plt.show()

        if emergency_control >= EMERGENCY_LIMIT:
            break

        x = x_tmp.copy()
        u = u_tmp.copy()

        x_cars.append(x)
        steps_cars.append(steps-num_overshoot)

        safeSet.append(current_safeSet)

        iter_costs.append(last_cost)
        print("Iteration cost:", last_cost)

        '''with open(f'costs_K_{K}.txt', 'a') as file:
            file.write(f"lap: {lap}\n")
            file.write(f"Iteration cost: {last_cost}\n")
            file.write(f"Computation time: {time.time() - lap_start_time}")
            file.write("\n")'''

    plt_cost = plt.subplot(1,2,1)
    plt_ss = plt.subplot(1,2,2)
    plt_cost.plot(iter_costs)
    plt_cost.set_title('Iteration costs')
    plt_ss.plot(samples_in_ss)
    plt_ss.set_title('Samples in safe set')
    plt.show()
   
    animation.animate_car(tr, x, u, x_pred_record, steps - num_overshoot, 10)
    
    return x_cars, steps_cars