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
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.patches import Polygon


class Track:

    # ========================
    # 1: Init Function
    # ========================
    def __init__(self, a, b, w):
        self.a  = a             #coefficients
        self.b  = b             #coefficients
        self.w  = w             # w: width dimension of the track
        
        self.s_low  = 0         # s_low: Minimum values of the abscissa
        self.s_high = 1         # s_high: limit value of abscissa
        
        # tau(s): Function that given the abscissa s return a (x,y) cartesian point
        self.tau = lambda s: [np.dot(a.T, [s**7, s**6, s**5, s**4, s**3, s**2, s, s**0]), np.dot(b.T, [s**7, s**6, s**5, s**4, s**3, s**2, s, s**0])] # tau: Vector of position
        
        # tau'(s): First Derivative wrt s parameter
        self.dtau = lambda s: [
            np.dot(a.T, [7*s**6, 6*s**5, 5*s**4, 4*s**3, 3*s**2, 2*s, 1, 0]),
            np.dot(b.T, [7*s**6, 6*s**5, 5*s**4, 4*s**3, 3*s**2, 2*s, 1, 0])
            ]
        
        # tau''(s): Second Derivative wrt s parameter
        self.ddtau = lambda s: [
            np.dot(a.T, [42*s**5, 30*s**4, 20*s**3, 12*s**2, 6*s, 2, 0, 0]),
            np.dot(b.T, [42*s**5, 30*s**4, 20*s**3, 12*s**2, 6*s, 2, 0, 0])
            ]
        
        self.dddtau = lambda s: [
            np.dot(a.T, [210*s**4, 120*s**3, 60*s**2, 24*s, 6, 0, 0, 0]),
            np.dot(b.T, [210*s**4, 120*s**3, 60*s**2, 24*s, 6, 0, 0, 0])
        ]
        
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
        #sp = lambda s: (dx(cs.fmod(s, self.s_high))**2 + dy(cs.fmod(s, self.s_high))**2)**(1/2)
        sp = lambda s: (dx(s)**2 + dy(s)**2)**(1/2)

        return sp
    # ========================
    # 3: Point Function
    # ========================
    def sn2xy(self, s, n):
    
        '''
        Function to get the position of the car 
        '''

        mod_s = s #% self.s_high  # It's needed because each lap we want the same tau(s) position 
        tau_x, tau_y = self.tau(mod_s) # Position coordinates on the track
        dtau_x, dtau_y = self.dtau(mod_s) # Velocity on the track
        sp = self.speed(s) # The speed on the track

        # Speed, dtau_y and dtau_x form a triangle on the track (speed is the hypotenuse, while the others are the cathetes)
        # In this part we compute the sine and cosine of the angles between velocity vector and x-axis
        sine = dtau_y / sp 
        cosine = dtau_x / sp

        # Position of the car computed through the error and angle
        pos = np.array([tau_x - n * sine, tau_y + n * cosine])

        return pos  
    
    def sxi2phi(self, s, xi):
        th = self.theta(s)

        return th + xi

    # ========================
    # 4: Theta Function
    # ========================
    def get_theta(self):
        '''
        Function to get the angle between tau_d and x-axis
        '''
        dx = lambda s: self.dtau(s)[0]
        dy = lambda s: self.dtau(s)[1]

        # Trigonometry formulas considering a triangle rectangle with
        # tau_d is the hypotenuse
        # dx, dy are the cathetes
        theta = lambda s: np.arctan2(dy(s), dx(s))

        return theta

    # ========================
    # 5: Track Function
    # ========================
    def get_track(self):

        '''
        Function to take track points, central line and borders
        '''
        t = np.linspace(self.s_low, self.s_high, 1000) # Array of 500 values between [s_low, s_high]
        tau_x, tau_y = self.tau(t) # Position of the car evaluated for all the discrete sequence t
        tau_inner_x = [] # Vector of x-coordinates of the inner border
        tau_inner_y = [] # Vector of y-coordinates of the inner border
        tau_outer_x = [] # Vector of x-coordinates of the outer border
        tau_outer_y = [] # Vector of y-coordinates of the outer border
    
    # Loop to compute all the points of the track
        for s in t:  

            tau_s_x = self.tau(s)[0] # x-coord of track's position at s value
            tau_s_y = self.tau(s)[1]  # y-coord of track's position at s value
            dtau_s_x = self.dtau(s)[0] # x-component of velocity at s value
            dtau_s_y = self.dtau(s)[1] # y-component of velocity at s value
            sp = self.speed(s) 

            sine = dtau_s_y / sp
            cosine = dtau_s_x / sp

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
        #k = lambda s: N(cs.fmod(s, self.s_high)) / D(cs.fmod(s,self.s_high))
        k = lambda s: N(s) / D(s)

        return k
    
    # ========================
    # 7:start functions
    # ========================
    def start(self):
        return np.array([self.tau(self.s_low)[0],self.tau(self.s_low)[1],self.theta(self.s_low)])
        
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

    def plot_k(self, plt):
        
        # Plot curvature
        t = np.linspace(self.s_low, self.s_high, 300)
        tau_x = []
        for x in t:
            tau_x.append(self.k(x).__float__())
        plt.plot(t, tau_x)
        plt.show()

