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
from matplotlib import pyplot as plt

# Our Functions
from utils import track_bez, track, animation
from controller import trivial_control, lmpc_control

EULER = 0
integration = EULER
# Taking a simple input from the user
print("\nSelect the track:")
user_input = input("1 -> Waterfall Track, 2 -> Tornado Circuit, 3 -> Monza: ")
scaling_factor = int(input("Choose the reducing factor (1 -> real track, 2 -> halved, so on): "))
lap_chosen = input("Choose the number of laps: ")
laps = int(lap_chosen)

if user_input == "1":
    #  Track - https://www.desmos.com/calculator/0mzpgjrlxt?lang=it
    scale = 126.899139037 / scaling_factor     # Factor of scale to real dimension
    w = 150 / scaling_factor         # Track width
    a = scale * np.array([-1880.78, 6243.74, -7155.46, 2782.74, 587.388, -732.109, 154.479, -0.17531]) # Coefficients of x points
    b = scale * np.array([5646.5, -17685.5, 20698.7, -11053.1, 2531.88, -140.358, 1.877, -4.89298]) # Coefficients of y points

    # retrieve model
    cascade_falls_track = track.Track(a, b, w)
    tr = cascade_falls_track
    print ("You've selected Waterfall Track!")

elif user_input == '2':
    scale = 126.899139037 / scaling_factor     # Factor of scale to real dimension
    w = 150 / scaling_factor         # Track width
    a = scale * np.array([7019.63, -27893.8, 44221.1, -35300.1, 14782.1, -3112.3, 283.435, -2.34658]) # Coefficients of x points
    b = scale * np.array([-4227.59, 14051.4, -17912.3, 11039.8, -3469.71, 520.798, -2.42002, -5.59871]) # Coefficients of y points

    # retrieve model 
    tornado_circuit= track.Track(a, b, w)
    tr = tornado_circuit
    print ("You've selected Tornado Circuit!")

elif user_input == "3":
    # Monza - Beziergon - https://www.desmos.com/calculator/2ykxxqoixs
    scale = 145.979885396 / scaling_factor     # Factor of scale to real dimension
    w = 150 / scaling_factor         # Track width

    x_b = scale *   np.array([1.27, -2.37, -3.689, -4.71, -6.263, -7.38, -8.07, -8.767, -6.8, -5.584, -2.18, -1.397, -0.682, 4.02, 6.7])
    y_b = scale *   np.array([-3.13, -3.158, -2.82, -3.397, -3.564, -0.48, 1.094, 3.028, 5.145, 2.8, -0.546, -0.817, -1.246, -1.25, -2.73])

    x = scale *     np.array([0.094, -2.795, -4.096, -5.1045, -6.87, -7.565, -8.288, -8.836, -6.242, -5.1, -1.956, -1.077, -0.047, 5.365, 4.72])
    y = scale *     np.array([-3.145, -3.167, -3.106, -3.424, -2.65, 0.674, 1.664, 3.577, 4.106, 2.02, -0.71, -0.955, -1.287, -1.256, -3.087])

    x_f = scale *   np.array([-0.69, -3.824, -4.482, -5.405, -7.59, -7.684, -8.474, -9, -5.87, -4.3, -1.62, -0.81, 0.704, 6.93, 3.926])
    y_f = scale *   np.array([-3.15, -3.234, -3.406, -3.46, -1.4, 1.282, 2.356, 4.798, 3.437, 1.09, -0.964, -1.073, -1.333, -1.17, -3.13])

    # retrieve model
    monza = track_bez.Track(x_b, y_b, x, y, x_f, y_f, w)
    tr = monza
    print ("You've selected Monza")

else:
    print("Your selection is not admissible! Exit...")
    exit()

# Plot track
tr.plot_track(plt)
plt.show()

'''
# Plot curvature
t = np.linspace(tr.s_low, tr.s_high, 300)
tau_x = []
for x in t:
    tau_x.append(tr.k(x).__float__())
plt.plot(t, tau_x)
plt.show()
'''

backSight = 2   # Number of previous iterations at which the model looks, (j - l) in the paper
safeSet = []

print("Building the safe set with trivial control...")
for i in range(backSight):
    currentSafeSet = trivial_control.run(tr, i, integration)
    safeSet.append(currentSafeSet)
    #for s in currentSafeSet:
    #   safeSet.append(s)

'''# Plot not touched elements of the safe set in black
x_q = []
y_q = []
for ss in safeSet[-1]:
    if ss[1] > tr.s_high:
        break
    point = tr.sn2xy(ss[1], ss[2])
    x_q.append(point[0])
    y_q.append(point[1])

plt.plot(tr.track[0], tr.track[1])
plt.plot(tr.track[2], tr.track[3], 'r--')
plt.plot(tr.track[4], tr.track[5], 'r--')
plt.scatter(x_q, y_q, color='k')
plt.show()'''

# LMPC
print("Starting LMPC control...")
x_cars, steps_cars = lmpc_control.run(tr, safeSet, backSight, integration, laps)

# Race
print ("Starting Race...")
max_step = max(steps_cars)
animation.race(tr, x_cars, max_step, 10)