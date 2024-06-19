# ========================
# ARC control via LMPC with error dynamics regression
# Authors: Luca Murra, Massimo Romano, Francesco Giarrusso
# Prof. Giuseppe Oriolo, Nicola Scianca
#
# This is a library of functions where we define the models
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

# ========================
# Section 2: Reduced Model 
# ========================
def get_reduced_model(track):

    '''
    Get car kinematic model, the reduced one
    to simplify the computation of the MPC
    Args:
        1. k_func: it's the curvature at the actual abscissa s
        2. speed_func: it's the speed of the track
    '''
    # Front and Rear axle distances (from the center of mass)
    Parameters = namedtuple('Parameters', ['Lf', 'Lr']) 
    p = Parameters(Lf = 1.105, Lr = 1.738)

    # Reduced State 
    # x = [v, s, n, xi] 
    # v: velocity applied at the center of mass
    # s: abscissa 
    # n: error between center of mass of the car and the track
    # xi: error between the orientation of the car and the orientation of the track
    v       = lambda x: x[0]
    s       = lambda x: x[1]
    n       = lambda x: x[2]
    xi      = lambda x: x[3]

    # Input: u =[a, delta_f]
    # a: Acceleration of the car
    # delta_f: steering angle of the car

    # Angle that v form with car axle (u[1] => Steering angle input)
    beta = lambda u: cs.atan(p.Lr / (p.Lr + p.Lf) * cs.tan(u[1])) 
    k = lambda x: track.k(s(x)) # Curvature at actual abscissa s (s(x) == x[1] == s)
    sp = lambda x: track.speed(s(x)) # Speed at actual abscissa s

    dphi = lambda x, u: v(x) / p.Lr * cs.sin(beta(u)) # Variation of steering of the car
    dv = lambda u: u[0] # Acceleration Input

    #  
    ds = lambda x, u: v(x) * cs.cos(xi(x) + beta(u)) / (1 - n(x) * k(x)) / sp(x) # Variation of the abscissa
    dn = lambda x, u: v(x) * cs.sin(xi(x) + beta(u)) # Variation of the distance to the track
    dxi = lambda x, u: dphi(x, u) - k(x) * ds(x, u) * sp(x) # Variation of the orientation wrt track

    # Vector of functions that we will use as f_tail in the get_car_kinematic_model function
    f = lambda x, u: cs.vertcat(dv(u), ds(x, u), dn(x, u), dxi(x, u)) 

    return (f, p) # Return also the parameters of the car

# ========================
# Section 3: Complete Model 
# ========================
def get_car_kinematic_model(track):
    '''
    Get car kinematic model (the complete one)
    Args:
        1. k_func: it's the curvature at the actual abscissa s
        2. speed_func: it's the speed of the track
    '''

    f_tail, p = get_reduced_model(track) 

    # Complete State
    # x = [X, Y, phi, beta, v, s, n, xi] = [X,Y, phi, beta, [reduced_state]]
    # X,Y: Position of the COM of the car
    # phi: angle of orientation of the car wrt x-axis
    # beta: angle between velocity (applied at COM) and the car axle
    phi     = lambda x: x[2]
    beta    = lambda x: x[3]
    v       = lambda x: x[4]

    dx = lambda x: v(x) * cs.cos(phi(x) + beta(x)) # x-component of the velocity vector
    dy = lambda x: v(x) * cs.sin(phi(x) + beta(x)) # y-component of the velocity vector
    dphi = lambda x: v(x) / p.Lr * cs.sin(beta(x)) # yaw rate (angular velocity omega_z)
    
    f = lambda x, u: cs.vertcat(dx(x), dy(x), dphi(x), 0, f_tail(x[4:], u))

    return (f, p)

def get_jacobians_here(track, dt, x, u):

    '''
    Get car kinematic model, the reduced one
    to simplify the computation of the MPC
    Args:
        1. k_func: it's the curvature at the actual abscissa s
        2. speed_func: it's the speed of the track
        x and u is the linearization point
    '''
    

    # Front and Rear axle distances (from the center of mass)
    Parameters = namedtuple('Parameters', ['Lf', 'Lr']) 
    p = Parameters(Lf = 1.105, Lr = 1.738)


    # Reduced State 
    # x = [v, s, n, xi] 
    # v: velocity applied at the center of mass
    # s: abscissa 
    # n: error between center of mass of the car and the track
    # xi: error between the orientation of the car and the orientation of the track
    v       = x[0]
    s       = x[1]
    n       = x[2]
    xi      = x[3]

    a       = u[0]
    delta   = u[1]

    # Input: u =[a, delta_f]
    # a: Acceleration of the car
    # delta_f: steering angle of the car

    # Angle that v form with car axle (u[1] => Steering angle input)
    beta = cs.atan(p.Lr / (p.Lr + p.Lf) * cs.tan(delta)) 
    k = track.k(s).__float__() # Curvature at actual abscissa s (s(x) == x[1] == s)
    sp = track.speed(s).__float__() # Speed at actual abscissa s

    den = 1-n*k
    cos = cs.cos(xi+beta)
    sin = cs.sin(xi+beta)

    df = track.dtau(s)[0].__float__()
    ddf = track.ddtau(s)[0].__float__()
    dddf = track.dddtau(s)[0].__float__()
    dg = track.dtau(s)[1].__float__()
    ddg = track.ddtau(s)[1].__float__()
    dddg = track.dddtau(s)[1].__float__()

    dsp_ds = (df*ddf + dg*ddg) / sp
    dk_ds = (df * dddg  - dddf * dg) / sp**3 - 3 * (df*ddg - ddf*dg) / sp**4 * dsp_ds

    ds_dk = n * v * cos/ den**2 * 1/sp
    ds_dsp = -v * cos / den * 1/sp**2

    dxi_dk = -n * v * cos / den**2 * k - v * cos / den
    
    A1 = [0, 0, 0, 0]
    A2 = [cos/(den*sp), ds_dk * dk_ds + ds_dsp * dsp_ds, k*v*cos/(den**2 * sp), -v*sin/(den*sp)]
    A3 = [sin, 0, 0, v*cos]
    A4 = [cs.sin(beta)/p.Lr - cos/den * k, dxi_dk * dk_ds, -k**2*v*cos/den**2, k*v*sin/den]
    
    A =  cs.DM.eye(4) + cs.DM([A1, A2, A3, A4]) * dt

    dbeta_ddelta =  p.Lr * (p.Lf + p.Lr) / ((cs.cos(delta) * (p.Lf+p.Lr))**2 + (p.Lr * cs.sin(delta))**2)
    
    B1 = [1, 0]
    B2 = [0, -v * sin * dbeta_ddelta/(den*sp)]
    B3 = [0, v * cos * dbeta_ddelta]
    B4 = [0, (v/p.Lr*cs.cos(beta) + k*v*sin/den)*dbeta_ddelta]
    
    B = cs.DM([B1, B2, B3, B4]) * dt

    dphi = v / p.Lr * cs.sin(beta) # Variation of steering of the car
    dv = a # Acceleration Input

    ds = v * cos / den * 1 / sp # Variation of the abscissa
    dn = v * sin # Variation of the distance to the track
    dxi = dphi - k * v * cos / den # Variation of the orientation wrt track

    C = x + cs.DM([dv, ds, dn, dxi])*dt - A @ x - B @ u

    return (A, B, C)

def get_jacobians_rk_here(track, dt, x_bar, u_bar):
        # Front and Rear axle distances (from the center of mass)
    Parameters = namedtuple('Parameters', ['Lf', 'Lr']) 
    p = Parameters(Lf = 1.105, Lr = 1.738)


    # Reduced State 
    # x = [v, s, n, xi] 
    # v: velocity applied at the center of mass
    # s: abscissa 
    # n: error between center of mass of the car and the track
    # xi: error between the orientation of the car and the orientation of the track
    v       = lambda x: x[0]
    s       = lambda x: x[1]
    n       = lambda x: x[2]
    xi      = lambda x: x[3]

    a       = u_bar[0]
    delta   = u_bar[1]

    # Input: u =[a, delta_f]
    # a: Acceleration of the car
    # delta_f: steering angle of the car

    # Angle that v form with car axle (u[1] => Steering angle input)
    beta = cs.atan(p.Lr / (p.Lr + p.Lf) * cs.tan(delta)) 
    k = lambda x: track.k(s(x)) # Curvature at actual abscissa s (s(x) == x[1] == s)
    sp = lambda x: track.speed(s(x)) # Speed at actual abscissa s

    den = lambda x: 1-n(x)*k(x)
    cos = lambda x: cs.cos(xi(x)+beta)
    sin = lambda x: cs.sin(xi(x)+beta)

    df = lambda x: track.dtau(s(x))[0]
    ddf = lambda x: track.ddtau(s(x))[0]
    dddf = lambda x: track.dddtau(s(x))[0]
    dg = lambda x: track.dtau(s(x))[1]
    ddg = lambda x: track.ddtau(s(x))[1]
    dddg = lambda x: track.dddtau(s(x))[1]

    dsp_ds = lambda x: (df(x)*ddf(x) + dg(x)*ddg(x)) / sp(x)
    dk_ds = lambda x: (df(x) * dddg(x)  - dddf(x) * dg(x)) / sp(x)**3 - 3 * (df(x)*ddg(x) - ddf(x)*dg(x)) / sp(x)**4 * dsp_ds(x)

    ds_dk = lambda x: n(x) * v(x) * cos(x)/ den(x)**2 * 1/sp(x)
    ds_dsp = lambda x: -v(x) * cos(x) / den(x) * 1/sp(x)**2

    dxi_dk = lambda x: -n(x) * v(x) * cos(x) / den(x)**2 * k(x) - v(x) * cos(x) / den(x)
    
    A1 = [0, 0, 0, 0]
    A2 = lambda x: [cos(x)/(den(x)*sp(x)), ds_dk(x) * dk_ds(x) + ds_dsp(x) * dsp_ds(x), k(x)*v(x)*cos(x)/(den(x)**2 * sp(x)), -v(x)*sin(x)/(den(x)*sp(x))]
    A3 = lambda x: [sin(x), 0, 0, v(x)*cos(x)]
    A4 = lambda x: [cs.sin(beta)/p.Lr - cos(x)/den(x) * k(x), dxi_dk(x) * dk_ds(x), -k(x)**2*v(x)*cos(x)/den(x)**2, k(x)*v(x)*sin(x)/den(x)]

    dphi = lambda x: v(x) / p.Lr * cs.sin(beta) # Variation of steering of the car
    dv = a # Acceleration Input

    ds = lambda x: v(x) * cos(x) / den(x) * 1 / sp(x) # Variation of the abscissa
    dn = lambda x: v(x) * sin(x) # Variation of the distance to the track
    dxi = lambda x: dphi(x) - k(x) * v(x) * cos(x) / den(x) # Variation of the orientation wrt track

    k1 = lambda x: cs.DM([dv, ds(x), dn(x), dxi(x)]).full().squeeze()
    k2 = k1(x_bar + 0.5 * dt * k1(x_bar))
    k3 = k1(x_bar + 0.5 * dt * k2)
    k4 = k1(x_bar + dt * k3)
    
    grad_k1 = lambda x: cs.DM([A1, A2(x), A3(x), A4(x)])
    grad_k2 = (1 + 0.5 * dt * grad_k1(x_bar)) * grad_k1(x_bar + 0.5* dt * k1(x_bar))
    grad_k3 = (1 + 0.5 * dt * grad_k2) * grad_k1(x_bar + 0.5 * dt * k2)
    grad_k4 = (1 + dt * grad_k3) * grad_k1(x_bar + dt * k3)

    A = cs.DM.eye(4) + (grad_k1(x_bar) + 2 * grad_k2 + 2 * grad_k3 + grad_k4) * dt/6

    dbeta_ddelta =  p.Lr * (p.Lf + p.Lr) / ((cs.cos(delta) * (p.Lf+p.Lr))**2 + (p.Lr * cs.sin(delta))**2)
    
    B1 = [1, 0]
    B2 = lambda x: [0, -v(x) * sin(x) * dbeta_ddelta/(den(x)*sp(x))]
    B3 = lambda x: [0, v(x) * cos(x) * dbeta_ddelta]
    B4 = lambda x: [0, (v(x)/p.Lr*cs.cos(beta) + k(x)*v(x)*sin(x)/den(x))*dbeta_ddelta]
    
    gradu_k1 = lambda x: cs.DM([B1, B2(x), B3(x), B4(x)])
    gradu_k2 = (1 + 0.5 * dt * gradu_k1(x_bar)) * gradu_k1(x_bar + 0.5 * dt * k1(x_bar))
    gradu_k3 = (1 + 0.5 * dt * gradu_k2) * gradu_k1(x_bar + 0.5 * dt * k2)
    gradu_k4 = (1 + dt * gradu_k3) * gradu_k1(x_bar + dt * k3)

    B = (gradu_k1(x_bar) + 2 * gradu_k2 + 2 * gradu_k3 + gradu_k4) * dt/6

    C = x_bar + (k1(x_bar) + 2*k2 + 2*k3 + k4)*dt/6 - A @ x_bar - B @ u_bar

    return (A, B, C)