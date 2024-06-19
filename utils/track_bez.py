# ========================
# ARC control via LMPC with error dynamics regression
# Authors: Luca Murra, Massimo Romano, Francesco Giarrusso
# Prof. Giuseppe Oriolo, Nicola Scianca
#
# This is a library of functions needed for the track 
# ========================

# ========================
# Section 1: Importing Libraries
# ========================
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
from matplotlib.patches import Polygon


class Track:

    # ========================
    # 1: Init Function
    # ========================
    def __init__(self, x_b, y_b, x, y, x_f, y_f, w):

        assert len(x_b) == len(x) == len(y) == len(y_b) == len(x_f) == len(y_f), "Points lists length must be equal"

        self.x_b  = x_b
        self.y_b  = y_b
        self.x  = x
        self.y = y
        self.x_f = x_f
        self.y_f = y_f

        self.w = w

        self.s_low = 0
        self.s_high = len(x)

        F = []
        G = []

        dF = []
        dG = []

        ddF = []
        ddG = []

        dddF = []
        dddG = []

        for i in range(len(x)):
            f = lambda s, i=i: (1-s)**3 * x[i] + 3*s * (1-s)**2 * x_f[i] + 3*s**2 * (1-s) * x_b[(i+1) % len(x)] + s**3 * x[(i+1) % len(x)]
            g = lambda s, i=i: (1-s)**3 * y[i] + 3*s * (1-s)**2 * y_f[i] + 3*s**2 * (1-s) * y_b[(i+1) % len(y)] + s**3 * y[(i+1) % len(y)]

            df = lambda s, i=i: -3 * (1-s)**2 * x[i] + 3*(1 - 4*s + 3*s**2) *x_f[i] + 3*(2 - 3*s) *s * x_b[(i+1) % len(x)] + 3*s**2 * x[(i+1) % len(x)]
            dg = lambda s, i=i: -3 * (1-s)**2 * y[i] + 3*(1 - 4*s + 3*s**2) *y_f[i] + 3*(2 - 3*s) *s * y_b[(i+1) % len(y)] + 3*s**2 * y[(i+1) % len(y)]

            ddf = lambda s, i=i: 6 * (1-s) * x[i] + 6*(-2 + 3*s) * x_f[i] + 6*(1 - 3*s) * x_b[(i+1) % len(x)] + 6*s * x[(i+1) % len(x)]
            ddg = lambda s, i=i: 6 * (1-s) * y[i] + 6*(-2 + 3*s) * y_f[i] + 6*(1 - 3*s) * y_b[(i+1) % len(y)] + 6*s * y[(i+1) % len(y)]

            dddf = lambda s, i=i: -6 * x[i] + 18 * x_f[i] - 18 * s * x_b[(i+1) % len(x)] + 6 * x[(i+1) % len(x)]
            dddg = lambda s, i=i: -6 * y[i] + 18 * y_f[i] - 18 * s * y_b[(i+1) % len(x)] + 6 * y[(i+1) % len(x)]

            F.append(f)
            G.append(g)

            dF.append(df)
            dG.append(dg)

            ddF.append(ddf)
            ddG.append(ddg)

            dddF.append(dddf)
            dddG.append(dddg)

        self.tau     = lambda s: np.array([F[int(s)](s - int(s)),    G[int(s)](s - int(s))])
        self.dtau    = lambda s: np.array([dF[int(s)](s - int(s)),   dG[int(s)](s - int(s))])
        self.ddtau   = lambda s: np.array([ddF[int(s)](s - int(s)),  ddG[int(s)](s - int(s))]) 
        self.dddtau  = lambda s: np.array([dddF[int(s)](s - int(s)),  dddG[int(s)](s - int(s))]) 
        
        # Define CasADi symbolic variable
        s = cs.MX.sym('s')

        # Floor operation using CasADi
        floored_s = cs.floor(s)

        # Define CasADi expressions for tau, dtau, and ddtau
        F_casadi = [cs.Function(f'F{i}', [s], [f(s - floored_s)]) for i, f in enumerate(F)]
        G_casadi = [cs.Function(f'G{i}', [s], [g(s - floored_s)]) for i, g in enumerate(G)]

        dF_casadi = [cs.Function(f'dF{i}', [s], [df(s - floored_s)]) for i, df in enumerate(dF)]
        dG_casadi = [cs.Function(f'dG{i}', [s], [dg(s - floored_s)]) for i, dg in enumerate(dG)]

        ddF_casadi = [cs.Function(f'ddF{i}', [s], [ddf(s - floored_s)]) for i, ddf in enumerate(ddF)]
        ddG_casadi = [cs.Function(f'ddG{i}', [s], [ddg(s - floored_s)]) for i, ddg in enumerate(ddG)]

        dddF_casadi = [cs.Function(f'dddF{i}', [s], [dddf(s - floored_s)]) for i, dddf in enumerate(dddF)]
        dddG_casadi = [cs.Function(f'dddG{i}', [s], [dddg(s - floored_s)]) for i, dddg in enumerate(dddG)]

        #self.tau = cs.Function('tau', [s], [cs.vertcat(self.select_casadi_function(F_casadi, floored_s, s),
        #                                                self.select_casadi_function(G_casadi, floored_s, s))])

        self.dtau = cs.Function('dtau', [s], [cs.vertcat(self.select_casadi_function(dF_casadi, floored_s, s),
                                                        self.select_casadi_function(dG_casadi, floored_s, s))])

        self.ddtau = cs.Function('ddtau', [s], [cs.vertcat(self.select_casadi_function(ddF_casadi, floored_s, s),
                                                            self.select_casadi_function(ddG_casadi, floored_s, s))])
        
        self.dddtau = cs.Function('dddtau', [s], [cs.vertcat(self.select_casadi_function(dddF_casadi, floored_s, s),
                                                            self.select_casadi_function(dddG_casadi, floored_s, s))])

        
        self.speed  =   self.get_curve_speed() # speed: function to retrieve speed of the track given abscissa s
        self.theta  =   self.get_theta()       # theta: angle between the track and x-axis
        self.track  =   self.get_track()       # track: array containing x,y coordinates for center of the track and borders
        self.k      =   self.get_k()           # k:     curvature given the abscissa s 


    # ========================
    # 2: Speed Function
    # ========================
    def get_curve_speed(self):
        '''
        Function to get the speed of the track
        '''
        dx = lambda s: self.dtau(s)[0] # velocity along x-axis
        dy = lambda s: self.dtau(s)[1] # velocity along y-axis
    
        # Speed function as the square root of the x-y components evaluated in actual abscissa s
        sp = lambda s: (dx(cs.fmod(s, self.s_high))**2 + dy(cs.fmod(s, self.s_high))**2)**(1/2)

        return sp

    # ========================
    # 3: Point Function
    # ========================
    def sn2xy(self, s, n):
    
        '''
        Function to get the position of the car 
        '''

        mod_s = s % self.s_high # It's needed because each lap we want the same tau(s) position (probably useless)
        #print(mod_s)
        tau_x, tau_y = self.tau(mod_s) # Position coordinates on the track
        dtau_x = self.dtau(mod_s)[0].__float__() # Velocity on the track
        dtau_y = self.dtau(mod_s)[1].__float__()
        sp = self.speed(mod_s).__float__() # The speed on the track

        # Speed, dtau_y and dtau_x form a triangle on the track (speed is the hypotenuse, while the others are the cathetes)
        # In this part we compute the sine and cosine of the angles between velocity vector and x-axis
        sine = dtau_y / sp 
        cosine = dtau_x / sp

        # Position of the car computed through the error and angle
        pos = np.array([tau_x - n * sine, tau_y + n * cosine])

        return pos  

    def sxi2phi(self, s, xi):
        th = self.theta(s).__float__()

        return th + xi
    
    # ========================
    # 4: Theta Function
    # ========================
    def get_theta(self):
        '''
        Function to get the angle between tau_d and x-axis
        '''
        dx = lambda s: self.dtau(s % self.s_high)[0]
        dy = lambda s: self.dtau(s % self.s_high)[1]

        # Trigonometry formulas considering a triangle rectangle with
        # tau_d is the hypotenuse
        # dx, dy are the cathetes
        theta = lambda s: np.arctan2(dy(s % self.s_high), dx(s % self.s_high))

        return theta

    # ========================
    # 5: Track Function
    # ========================
    def get_track(self):

        '''
        Function to take track points, central line and borders
        '''
        t = np.linspace(0, self.s_high -0.001, 500) # Array of 500 values between [s_low, s_high]

        tau_x = []
        tau_y = []
        tau_inner_x = [] # Vector of x-coordinates of the inner border
        tau_inner_y = [] # Vector of y-coordinates of the inner border
        tau_outer_x = [] # Vector of x-coordinates of the outer border
        tau_outer_y = [] # Vector of y-coordinates of the outer border
    
    # Loop to compute all the points of the track
        for s in t:  

            tau_s_x = self.tau(s)[0].__float__() # x-coord of track's position at s value
            tau_s_y = self.tau(s)[1].__float__()  # y-coord of track's position at s value
            dtau_s_x = self.dtau(s)[0].__float__() # x-component of velocity at s value
            dtau_s_y = self.dtau(s)[1].__float__() # y-component of velocity at s value
            sp = self.speed(s).__float__() 

            sine = dtau_s_y / sp
            cosine = dtau_s_x / sp

            tau_x.append(tau_s_x)
            tau_y.append(tau_s_y)

            # These are the formula used to compute the points of the inner border
            # These are equivalent to the formula of position in sn2xy() where n=w/2
        
            # Inner Border Computation
            tau_inner_x.append(tau_s_x - self.w/2 * sine)
            tau_inner_y.append(tau_s_y + self.w/2 * cosine)

            # Outer Border Computation
            tau_outer_x.append(tau_s_x + self.w/2 * sine)
            tau_outer_y.append(tau_s_y - self.w/2 * cosine)

        return [tau_x, tau_y, tau_inner_x, tau_inner_y, tau_outer_x, tau_outer_y]

    # ========================
    # 6: Curvature Function
    # ========================
    def get_k(self):
        '''
        Function to get the curvature of the track
        Link to the formula:
        https://courses.lumenlearning.com/calculus3/chapter/curvature/
        '''
        dx = lambda x: self.dtau(x)[0]
        dy = lambda x: self.dtau(x)[1]
        ddx = lambda x: self.ddtau(x)[0]
        ddy = lambda x: self.ddtau(x)[1]

        N = lambda x: dx(x) * ddy(x) - ddx(x) * dy(x) # Numerator of the formula
        D = lambda x: (dx(x)**2 + dy(x)**2)**(3/2) # Denominator of the formula

        # Curvature formula
        k = lambda s: N(cs.fmod(s, self.s_high)) / D(cs.fmod(s, self.s_high))

        return k
    
    # ========================
    # 7:start functions
    # ========================
    def start(self):
        return np.array([self.tau(0)[0].__float__(), self.tau(0)[1].__float__() , self.theta(0).__float__()])

    # Function to select the appropriate element based on the floored value of s
    def select_casadi_function(self, f_list, idx, s, c = 0):
        if c == len(f_list) -1:
            return f_list[-1](s)
        else:
            return cs.if_else(idx == c, f_list[c](s), self.select_casadi_function(f_list, idx, s, c = c+1))
        
    def plot_track(self, plt):
        # Convert track lists to numpy arrays
        track0 = np.array(self.track[0])
        track1 = np.array(self.track[1])
        track2 = np.array(self.track[2])
        track3 = np.array(self.track[3])
        track4 = np.array(self.track[4])
        track5 = np.array(self.track[5])

        plt.plot(self.track[0], self.track[1], 'w--')
        plt.plot(self.track[2], self.track[3], 'k')
        plt.plot(self.track[4], self.track[5], 'k')

        # Draw the track background as a filled polygon
        inner_border = np.column_stack((track2, track3))
        outer_border = np.column_stack((track4, track5))
        track_polygon = np.vstack((inner_border, outer_border[::-1]))  # Combine inner and outer borders

        plt.gca().add_patch(Polygon(track_polygon, closed=True, color='gray', alpha=0.5))  # Add polygon patch
        plt.gca().set_aspect('equal')  # Set equal aspect ratio for the plot
