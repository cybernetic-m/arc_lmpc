# ========================
# ARC control via LMPC with error dynamics regression
# Authors: Luca Murra, Massimo Romano, Francesco Giarrusso
# Prof. Giuseppe Oriolo, Nicola Scianca
#
# This is a library of functions needed for the animation 
# ========================

import os
import numpy as np 
import math
import casadi as cs
#import animation
import time
from collections import namedtuple
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.transforms import Affine2D
from matplotlib.patches import Polygon


Lf = 10*1.105
Lr = 10*1.738 /2

# Function to get the directory of the current script
def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

# Function to get rotated triangle for car visualization
def get_car_polygon(p, phi):
    frontx, fronty = (p[0] + Lf * np.cos(phi), p[1] + Lf * np.sin(phi))
    rear1 = np.array([-Lr, +Lr])
    rear2 = np.array([-Lr, -Lr])
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    rear1xy = R @ rear1
    rear2xy = R @ rear2
    return np.vstack(([frontx, fronty], p + rear1xy, p + rear2xy))

def animate_car(tr, x, u, x_pred_record, steps, intervals):
   
    # Convert track lists to numpy arrays
    track0 = np.array(tr.track[0])
    track1 = np.array(tr.track[1])
    track2 = np.array(tr.track[2])
    track3 = np.array(tr.track[3])
    track4 = np.array(tr.track[4])
    track5 = np.array(tr.track[5])

    def animate(i):
        ax_large.cla()
        ax_large_zoom.cla()
        ax_large.plot(tr.track[0], tr.track[1], 'w--')
        ax_large_zoom.plot(tr.track[0], tr.track[1], 'w--')

        # Draw the track background as a filled polygon
        inner_border = np.column_stack((track2, track3))
        outer_border = np.column_stack((track4, track5))
        track_polygon = np.vstack((inner_border, outer_border[::-1]))  # Combine inner and outer borders
        ax_large.add_patch(Polygon(track_polygon, closed=True, color='black', alpha=0.5))
        ax_large_zoom.add_patch(Polygon(track_polygon, closed=True, color='black', alpha=0.5))

        s = x[5, i] 
        n = x[6, i]
        xi = x[7,i]
        p = tr.sn2xy(s, n)
        phi = tr.sxi2phi(s, xi)
        
        '''
        # Draw predicted trajectory
        N_pred = x_pred_record[0].shape[1]
        p_pred = np.zeros((2, N_pred))
        for j in range(N_pred):
            s_pred = x_pred_record[i][1, j]
            n_pred = x_pred_record[i][2, j]
            # s, n, tau, tau_d, s_high, speed
            p_pred[:, j] = tr.sn2xy(s_pred, n_pred)
        ax_large.plot(p_pred[0,:], p_pred[1,:], color='orange', zorder=1000)
        ax_large_zoom.plot(p_pred[0,:], p_pred[1,:], color='orange', zorder=1000)
        '''
        
        xy = get_car_polygon(p, phi)
        
        ax_large.add_patch(patches.Polygon(xy, edgecolor='black', facecolor='red', fill=True, zorder=10))
        ax_large_zoom.add_patch(patches.Polygon(xy, edgecolor='black', facecolor='red', fill=True, zorder = 10))

    
        ax_large_zoom.axis((p[0]- 200, p[0]+ 200, p[1]- 200, p[1]+ 200))

        
        '''
        ax_small11.cla()
        ax_small11.set_xlim(0, steps)
        ax_small11.plot(x[4, :i].T)
        ax_small11.legend(['v'], loc = 'lower center', ncol=2)

        ax_small21.cla()
        ax_small21.set_xlim(0, steps)
        ax_small21.plot(x[2:4, :i].T)
        ax_small21.legend(['phi', 'beta'], loc = 'lower center', ncol=2)

        ax_small12.cla()
        ax_small12.set_xlim(0, steps)
        #ax_small12.plot(x[6:, :i].T)
        
        k_x = []
        for s in x[5, :i]:
            k_x.append(tr.k(s).__float__())
        ax_small12.plot(k_x)
        
        ax_small12.legend(['k'], loc = 'lower center', ncol=2)

        ax_small22.cla()
        ax_small22.axis((0, steps, a_min*1.1, a_max*1.1))
        ax_small22.plot(u[:, :i].T)
        ax_small22.legend(['a', 'delta'], loc = 'lower center', ncol=2)
        '''

    # display
    fig = plt.gcf()
    fig.set_size_inches(15, 8)

    grid = GridSpec(2, 3)
    ax_large = plt.subplot(grid[0,:]) #[0, 0]
    ax_large_zoom = plt.subplot(grid[1, :]) # [1, 0]
    '''
    ax_small11 = plt.subplot(grid[0, 1])
    ax_small21 = plt.subplot(grid[1, 1])
    ax_small12 = plt.subplot(grid[0, 2])
    ax_small22 = plt.subplot(grid[1, 2])
    '''

    position = x[0:2, :]
    x_min = min(position[0, :])*1.1
    x_max = max(position[0, :])*1.1
    y_min = min(position[1, :])*1.1
    y_max = max(position[1, :])*1.1
    ax_large.axis((x_min, x_max, y_min, y_max))
    ax_large.set_aspect('equal')

    ax_large_zoom.set_aspect('equal')

    # x = [X, Y, phi, beta, v, s, n, xi]
    #print('%2s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s' % ('i','x', 'y', 's','dx', 'dy', 'ds', 'n', 'k', 'phi', 'beta', 'xi', 'theta', 'exp. xi', 'err'))


    ani = FuncAnimation(plt.gcf(), animate, frames=steps, repeat=True, interval=intervals)
    plt.show()

def race(tr, x_cars, steps, intervals): 

    script_dir = get_script_directory()
    car_img_path = os.path.join(script_dir, 'images/car.png')
    start_flag_path = os.path.join(script_dir, 'images/start_flag.png')

    car_img = mpimg.imread(car_img_path)  # Load the car image
    car_semi_height = tr.w/3  # Car image semi height
    car_aspect_ratio = car_img.shape[0] / car_img.shape[1]

    # Convert track lists to numpy arrays
    track0 = np.array(tr.track[0])
    track1 = np.array(tr.track[1])
    track2 = np.array(tr.track[2])
    track3 = np.array(tr.track[3])
    track4 = np.array(tr.track[4])
    track5 = np.array(tr.track[5])

    # Calculate the track dimensions
    x_min = min(track0.min(), track2.min(), track4.min()) * 1.1
    x_max = max(track0.max(), track2.max(), track4.max()) * 1.1
    y_min = min(track1.min(), track3.min(), track5.min()) * 1.1
    y_max = max(track1.max(), track3.max(), track5.max()) * 1.1

    start_flag = mpimg.imread(start_flag_path)  # Load the start flag
    flag_aspect_ratio = start_flag.shape[0] / start_flag.shape[1]
    
    # Get the coordinates for the start flag
    start_flag_center = tr.tau(0)
    
    # Assume start_flag is square for simplicity; adjust as necessary
    flag_semi_height = tr.w/2 # Adjust this based on your actual image size

    # Calculate the extent for the flag image
    start_flag_extent = [
        start_flag_center[0] - flag_semi_height / flag_aspect_ratio,  # left
        start_flag_center[0] + flag_semi_height / flag_aspect_ratio,  # right
        start_flag_center[1] - flag_semi_height,  # bottom
        start_flag_center[1] + flag_semi_height   # top
    ]

    def animate(i):
        ax_large.cla()

        ax_large.plot(tr.track[0], tr.track[1], 'w--')

        flag_rotation = np.degrees(tr.theta(0))
        # Create rotation transformation for the start flag
        trans_data_flag = Affine2D().rotate_deg_around(start_flag_center[0], start_flag_center[1], flag_rotation) + ax_large.transData

        ax_large.imshow(start_flag, extent=start_flag_extent, origin='upper', zorder=100, transform=trans_data_flag)

        # Draw the track background as a filled polygon
        inner_border = np.column_stack((track2, track3))
        outer_border = np.column_stack((track4, track5))
        track_polygon = np.vstack((inner_border, outer_border[::-1]))  # Combine inner and outer borders

        ax_large.add_patch(Polygon(track_polygon, closed=True, color='black', alpha=0.5))
        
        #indices = np.round(np.linspace(0, len(x_cars)-1, num=5)).astype(int)
        indices = [0, 2, 7, 29]

        for car in indices:
            x = x_cars[car]
            s = x[5, i]
                
            ax_large.axis((x_min, x_max, y_min, y_max))
 
            if s < tr.s_high:
                n = x[6, i]
                xi = x[7, i]
                p = tr.sn2xy(s, n)
                phi = tr.sxi2phi(s, xi)
                label = car + 1

                trans_data = Affine2D().rotate_deg_around(p[0], p[1], np.degrees(phi)) + ax_large.transData
                img_extent = [
                    p[0] - car_semi_height / car_aspect_ratio,  # left
                    p[0] + car_semi_height / car_aspect_ratio,  # right
                    p[1] - car_semi_height,     # bottom
                    p[1] + car_semi_height      # top
                ]
                
                ax_large.imshow(car_img, extent=img_extent, origin='lower', zorder=101, transform=trans_data)

                ax_large.axis((x_min, x_max, y_min, y_max))

                ax_large.annotate(str(label), xy=(p[0] + tr.w / 4, p[1] + tr.w / 4), color='black', weight='bold', fontsize=15, ha='center', va='center', zorder=2)
                
            else:
                x[5, :] = tr.s_high + 1
                break
                
    # Display
    fig = plt.gcf()
    fig.set_size_inches(15, 8)

    grid = GridSpec(2, 3)
    ax_large = plt.subplot(grid[:, :])
    
    ax_large.set_aspect('equal')
    
    ani = FuncAnimation(plt.gcf(), animate, frames=steps, repeat=False, interval=intervals)
    plt.show()
   
    

    