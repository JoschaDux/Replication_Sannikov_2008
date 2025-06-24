#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 21:33:57 2025

@author: joschaduchscherer
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Define model parameters
class Model:
    def __init__(self):
        self.delta = 0.1 # Time-Preference Rate
        self.sigma = 1 # Output Volatility
        self.w_oo = 0.1 # Outside Option
        self.cost = 0.0341 # Replacement Cost
        self.theta = 1.5  # Promotion Productivity
        self.K = 0.1 # Promotion Cost
        self.w_oo_p = 0.2 # Outside Option Promotion

#Define grid parameters
class Grid:
    def __init__(self):
        self.wmin = 0
        self.wmax = 2.0
        self.tmin = 0
        self.tmax = 100
        self.Nw = 100
        self.Nw_p = 200
        self.Nt = 100000
        self.dt = (self.tmax - self.tmin) / self.Nt
        self.dw = (self.wmax - self.wmin) / self.Nw
        self.dw_p  = (self.wmax - self.wmin) / self.Nw_p
        self.nw = np.arange(self.Nw + 1)
        self.nw_p = np.arange(self.Nw_p + 1)
        self.nt = np.arange(self.Nt + 1)

model = Model()
grid = Grid()

# Define solver for trigonal matrix system
def trisolver(a_eq, b_eq, c_eq, d_eq, value_oo):
    
    size = len(a_eq)
    output = np.zeros(size)
    execution = np.zeros(size, dtype=int)
    
    c_prime = np.zeros(size)
    d_prime = np.zeros(size)
    
    
    # Forrward Substitution
    c_prime[0] = c_eq[0]/b_eq[0]
    d_prime[0] = d_eq[0]/b_eq[0]
    
    for i in range(1, size - 1):  
        denom = b_eq[i] - a_eq[i] * c_prime[i - 1]
        c_prime[i] = c_eq[i] / denom
        d_prime[i] = (d_eq[i] - a_eq[i] * d_prime[i - 1]) / denom
    
    d_prime[size-1] = (d_eq[size-1]-a_eq[size-1]*d_prime[size-2])/(b_eq[size-1]-a_eq[size-1]*c_prime[size-2])
    
    # Backward Substitution
    
    # Last element, compare to value of the outside option in the corresponding scenario
    last_values = [d_prime[size - 1], value_oo[size - 1]]
    output[size - 1] = max(last_values)
    execution[size - 1] = np.argmax(last_values)  # 0 if d_prime assigned, 1 if value_oo assigned
    
    for k in range(size - 2, -1, -1):
        values = [d_prime[k] - c_prime[k] * output[k + 1], value_oo[k]]
        output[k] = max(values)
        execution[k] = np.argmax(values)
    
    return output, execution

# Policy function for first, second and and fourth scenario
def policy_sen_I_II_p(F_ns, ret, w, theta):
    
    # Calculate grid spacing
    dw = w[1]-w[0]
    
    # Calculate numerical derivative
    F_ns_w = (np.roll(F_ns, -1) - np.roll(F_ns, 1)) / (2 * dw)
    F_ns_ww = (np.roll(F_ns, -1)-2*F_ns +np.roll(F_ns, 1)) / (dw**2)
    
    # Calculate optimal effort given by FOC
    a_FOC = -0.4-theta/(F_ns_w+F_ns_ww*model.delta*model.sigma**2/(theta**2))
    
    # Linear extrapolation of derivative a_FOC at lower bound
    slope_lb_a = (a_FOC[1] - a_FOC[2]) / (w[1] - w[2])
    intercept_lb_a= a_FOC[1] - slope_lb_a * w[1]
    a_FOC[0] = slope_lb_a*w[0]+intercept_lb_a
    
    a_constraint = a_FOC > 0
    a = a_constraint * a_FOC 
    
    c_constraint = F_ns_w < 0
    c = ((-0.5 * F_ns_w) ** 2) * c_constraint
    
    # Adjust for retirement
    c_ret = w ** 2
    a = (1 - ret) * a
    c = (1 - ret) * c + ret * c_ret
    c[0] = 0
    c[-1] = w[-1] ** 2
    
    a[-1] = 0
    
    return c, a

# Policy function for replacement scneario
def policy_sen_III(F_ns, replace, index, w):
    
    # Calculate grid spacing
    dw = w[1]-w[0]
    
    # Calculate numerical derivative
    F_ns_w = (np.roll(F_ns, -1) - np.roll(F_ns, 1)) / (2 * dw)
    F_ns_ww = (np.roll(F_ns, -1)-2*F_ns +np.roll(F_ns, 1)) / (dw**2)
    
    # Calculate optimal effort given by FOC
    a_FOC = -0.4-1/(F_ns_w+F_ns_ww*model.delta*model.sigma**2)
    
    # Linear extrapolation of derivative F_w at lower bound
    slope_lb_F = (F_ns_w[1] - F_ns_w[2]) / (w[1] - w[2])
    intercept_lb_F= F_ns_w[1] - slope_lb_F * w[1]
    F_ns_w[0] = slope_lb_F*w[0]+intercept_lb_F
    
    # Linear extrapolation of derivative F_w at upper bound
    slope_ub_F = (F_ns_w[-2] - F_ns_w[-3]) / (w[-2] - w[-3])
    intercept_ub_F = F_ns_w[-2] - slope_ub_F * w[-2]
    F_ns_w[-1] = slope_ub_F*w[-1]+intercept_ub_F
    
    # Linear extrapolation of derivative a_FOC at lower bound
    slope_lb_a = (a_FOC[1] - a_FOC[2]) / (w[1] - w[2])
    intercept_lb_a= a_FOC[1] - slope_lb_a * w[1]
    a_FOC[0] = slope_lb_a*w[0]+intercept_lb_a
    
    # Linear extrapolation of derivative a_FOC at upper bound
    slope_ub_a = (a_FOC[-2] - a_FOC[-3]) / (w[-2] - w[-3])
    intercept_ub_a = a_FOC[-2] - slope_ub_a * w[-2]
    a_FOC[-1] = slope_ub_a*w[-1]+intercept_ub_a
    
    a_constraint = a_FOC > 0
    a = a_constraint * a_FOC 
    
    a_0 = a*(1-replace)
    a_1 = a[index]
    
    c_constraint = F_ns_w < 0
    c = ((-0.5 * F_ns_w) ** 2) * c_constraint
    
    # Adjust for retirement
    c_replace = w** 2
    
    c_0 = (1-replace)*c+c_replace*replace
    c_1 = c_0[index]
    
    a_ges = (1 - replace) * a_0+a_1*replace
    c_ges = (1 - replace) * c_0 + replace*(c_0+c_1)
    
    return c_ges, a_ges, c_0, a_0, c_1, a_1

# Policy function for fourth scenario (unpromoted agent)
def policy_sen_IV(F_ns, exe, a_p_stack, c_p_stack, w):

    # Calculate grid spacing
    dw = w[1]-w[0]
    
    # Calculate numerical derivative
    F_ns_w = (np.roll(F_ns, -1) - np.roll(F_ns, 1)) / (2 * dw)
    F_ns_ww = (np.roll(F_ns, -1)-2*F_ns +np.roll(F_ns, 1)) / (dw**2)
    
    # Calculate optimal effort given by FOC
    a_FOC = -0.4-1/(F_ns_w+F_ns_ww*model.delta*model.sigma**2)
    
    # Linear extrapolation of derivative F_w at lower bound
    slope_lb_F = (F_ns_w[1] - F_ns_w[2]) / (w[1] - w[2])
    intercept_lb_F= F_ns_w[1] - slope_lb_F * w[1]
    F_ns_w[0] = slope_lb_F*w[0]+intercept_lb_F
    
    # Linear extrapolation of derivative F_w at upper bound
    slope_ub_F = (F_ns_w[-2] - F_ns_w[-3]) / (w[-2] -w[-3])
    intercept_ub_F = F_ns_w[-2] - slope_ub_F * w[-2]
    F_ns_w[-1] = slope_ub_F*w[-1]+intercept_ub_F
    
    # Linear extrapolation of derivative a_FOC at lower bound
    slope_lb_a = (a_FOC[1] - a_FOC[2]) / (w[1] - w[2])
    intercept_lb_a= a_FOC[1] - slope_lb_a * w[1]
    a_FOC[0] = slope_lb_a*w[0]+intercept_lb_a
    
    # Linear extrapolation of derivative a_FOC at upper bound
    slope_ub_a = (a_FOC[-2] - a_FOC[-3]) / (w[-2] - w[-3])
    intercept_ub_a = a_FOC[-2] - slope_ub_a * w[-2]
    a_FOC[-1] = slope_ub_a*w[-1]+intercept_ub_a
    
    a_constraint = a_FOC > 0
    a = a_constraint * a_FOC 
    
    c_ret = w**2
    
    c_constraint = F_ns_w < 0
    c = ((-0.5 * F_ns_w) ** 2) * c_constraint
    
    a = (exe == 0) * a + (exe == 1) * a_p_stack
    c = (exe == 0) * c + (exe == 1) * c_p_stack + (exe == 2) * c_ret
    c[0] = 0
    
    return c, a

def coefficients(c, a, w, theta):
    
    # Calculate grid spacing
    dw = w[1]-w[0]
    
    drift = model.delta*(w-np.sqrt(c)+0.5*a**2+0.4*a)
    vola_sq = model.delta**2*model.sigma**2*(a+0.4)**2/(theta**2)
    
    coe_a = 0.5*grid.dt*drift/dw-0.5*grid.dt*vola_sq/(dw**2)
    coe_b = 1+grid.dt*model.delta+vola_sq*grid.dt/(dw**2)
    coe_c = -0.5*grid.dt*drift/dw-0.5*grid.dt*vola_sq/(dw**2)
    
    return coe_a, coe_b, coe_c

start_time = time.time()
# Initialize value function and policies
F_I = np.zeros((grid.Nw + 1, grid.Nt+1))
c_I = np.zeros_like(F_I)
a_I = np.zeros_like(F_I)
ret_I = np.zeros((grid.Nw + 1, grid.Nt+1), dtype=int)

F_II = np.zeros_like(F_I)
c_II = np.zeros_like(F_I)
a_II = np.zeros_like(F_I)
ret_II = np.zeros((grid.Nw + 1, grid.Nt+1), dtype=int)

F_III = np.zeros_like(F_I)
c_III_ges = np.zeros_like(F_I)
c_III_0 = np.zeros_like(F_I)
c_III_1 = np.zeros_like(F_I)
a_III_ges = np.zeros_like(F_I)
a_III_0 = np.zeros_like(F_I)
a_III_1 = np.zeros_like(F_I)
rep_III = np.zeros((grid.Nw + 1, grid.Nt+1), dtype=int)
D = np.zeros((1, grid.Nt+1))

F_IV = np.zeros((grid.Nw_p + 1, grid.Nt+1))
c_IV = np.zeros_like(F_IV)
a_IV = np.zeros_like(F_IV)
exe_IV = np.zeros((grid.Nw_p + 1, grid.Nt+1), dtype=int)

F_IV_p = np.zeros((grid.Nw_p + 1, grid.Nt+1))
c_IV_p = np.zeros_like(F_IV_p)
a_IV_p = np.zeros_like(F_IV_p)
ret_IV_p = np.zeros((grid.Nw_p + 1, grid.Nt+1), dtype=int)

# Get grid axis in each scenario
w_I = grid.wmin+grid.dw*grid.nw
w_II = model.w_oo+grid.dw*grid.nw
w_IV = grid.wmin+grid.dw_p*grid.nw_p
w_IV_p = model.w_oo_p+grid.dw_p*grid.nw_p

# Value function and policies at maturity Scenario I
F_I[:, -1] = -(w_I)**2
F_I_ret = -(w_I)**2
F_I_ns_old = F_I[:, -1].copy()
ret_I_ns_old = ret_I[:, -1].copy()

c_I_ns, a_I_ns = policy_sen_I_II_p(F_I[:, -1], ret_I[:, -1], w_I, 1.0)
c_I[:, -1] = c_I_ns
c_I_ns_old = c_I_ns.copy()
a_I[:, -1] = a_I_ns
a_I_ns_old = a_I_ns.copy()

# Value function and policies at maturity Scenario II
F_II[:, -1] = -(w_II)**2
F_II_ns_old = F_II[:, -1].copy()
ret_II_ns_old = ret_II[:, -1].copy()
c_II_ns, a_II_ns = policy_sen_I_II_p(F_II[:, -1], ret_II[:, -1], w_II, 1.0)
c_II[:, -1] = c_II_ns
c_II_ns_old = c_II_ns.copy()
a_II[:, -1] = a_II_ns
a_II_ns_old = a_II_ns.copy()

# Value function and policies at maturity Scenario III
F_III[:, -1] = -(w_I)**2
F_III_ns_old = F_III[:, -1].copy()
rep_III_ns_old = rep_III[:, -1].copy()
pos_replace = 0
D[-1] = -model.cost
D_ns_old = -model.cost # Replacement profit
c_III_ns_ges, a_III_ns_ges, c_III_ns_0, a_III_ns_0, c_III_ns_1, a_III_ns_1 = policy_sen_III(F_III[:, -1], rep_III[:, -1], pos_replace, w_I)
c_III_ges[:, -1] = c_III_ns_ges
c_III_ns_ges_old = c_III_ns_ges.copy()
c_III_0[:, -1] = c_III_ns_0
c_III_ns_0_old = c_III_ns_0.copy()
c_III_1[:, -1] = c_III_ns_1
c_III_ns_1_old = c_III_ns_1.copy()
a_III_ges[:, -1] = a_III_ns_ges
a_III_ns_ges_old = a_III_ns_ges.copy()
a_III_0[:, -1] = a_III_ns_0
a_III_ns_0_old = a_III_ns_0.copy()
a_III_1[:, -1] = a_III_ns_1
a_III_ns_1_old = a_III_ns_1.copy()

# Get index of grid-p in grid of unpromoted agent
grid_index_l = np.where(np.abs(w_IV - model.w_oo_p) < 0.001)[0]
grid_index_u = np.where(np.abs(w_IV_p- grid.wmax) < 0.001)[0]
grid_index_l = int(grid_index_l[0])
grid_index_u = int(grid_index_u[0])

# Value function and policies at maturity Scenario IV
F_IV[:, -1] = -(w_IV)**2
F_IV_ns_old = F_IV[:, -1].copy()
exe_IV_ns_old = exe_IV[:, -1].copy()

F_IV_p[:, -1] = -(w_IV_p)**2
F_IV_p[0, -1] = 0
F_IV_p_ns_old = F_IV_p[:, -1].copy()
ret_IV_p_ns_old = ret_IV_p[:, -1].copy()
c_IV_p_ns, a_IV_p_ns = policy_sen_I_II_p(F_IV_p[:, -1], ret_IV_p[:, -1], w_IV_p, 1.5)
c_IV_p[:, -1] = c_IV_p_ns
c_IV_p_ns_old= c_IV_p_ns.copy()
a_IV_p[:, -1] = a_IV_p_ns
a_IV_p_ns_old= a_IV_p_ns.copy()

c_p_stack_ns = np.concatenate([np.zeros((grid_index_l)), c_IV_p_ns[:grid_index_u+1]])
a_p_stack_ns = np.concatenate([np.zeros((grid_index_l)), a_IV_p_ns[:grid_index_u+1]])

c_IV_ns, a_IV_ns = policy_sen_IV(F_IV[:, -1], exe_IV[:, -1], a_p_stack_ns, c_p_stack_ns, w_IV)
c_IV[:, -1] = c_IV_ns
c_IV_ns_old = c_IV_ns.copy()
a_IV[:, -1] = a_IV_ns
a_IV_ns_old = a_IV_ns.copy()

# Retirement values on both girds
F_ret_IV = -w_IV**2
F_ret_IV_p = -w_IV_p**2


for j in range(grid.Nt - 1, -1, -1):
    print(j)
    
    # Define vector for values at next time step
    F_I_ns = np.zeros_like(F_I_ns_old)
    ret_I_ns = np.zeros_like(ret_I_ns_old)
    
    F_II_ns = np.zeros_like(F_II_ns_old)
    ret_II_ns = np.zeros_like(ret_II_ns_old)
    
    F_III_ns = np.zeros_like(F_III_ns_old)
    rep_III_ns = np.zeros_like(rep_III_ns_old)
    
    F_IV_ns = np.zeros_like(F_IV_ns_old)
    exe_IV_ns = np.zeros_like(exe_IV_ns_old)
    
    F_IV_p_ns = np.zeros_like(F_IV_p_ns_old)
    ret_IV_p_ns = np.zeros_like(ret_IV_p_ns_old)
    
    # Set boundary values for F_I
    F_lower_I = 0
    F_upper_I= -w_I[grid.Nw] ** 2
    
    # Set boundary values for F_II
    F_lower_II = 0
    F_upper_II= -w_II[grid.Nw] ** 2
    
    # Set boundary values for F_III
    F_lower_III = D_ns_old
    F_upper_III= -w_I[grid.Nw] ** 2+D_ns_old
    
    # Set boundary values for F_IV_p
    F_lower_IV_p = 0
    F_upper_IV_p= -w_IV_p[grid.Nw_p] ** 2
    
    # Get the coefficients
    coe_a_I, coe_b_I, coe_c_I = coefficients(c_I_ns_old, a_I_ns_old, w_I, 1.0)
    coe_a_II, coe_b_II, coe_c_II = coefficients(c_II_ns_old, a_II_ns_old, w_II, 1.0)
    coe_a_III, coe_b_III, coe_c_III = coefficients(c_III_ns_ges_old, a_III_ns_ges_old, w_I, 1.0)
    coe_a_IV, coe_b_IV, coe_c_IV = coefficients(c_IV_ns_old, a_IV_ns_old, w_IV, 1.0)
    coe_a_IV_p, coe_b_IV_p, coe_c_IV_p = coefficients(c_IV_p_ns_old, a_IV_p_ns_old, w_IV_p, 1.5)

    # Construct elements for equation system and adjust the size
    d_eq_I = F_I_ns_old[1:grid.Nw]+grid.dt * model.delta * (a_I_ns_old[1:grid.Nw] - c_I_ns_old[1:grid.Nw])
    d_eq_I[0] = d_eq_I[0]-coe_a_I[1] * F_lower_I
    d_eq_I[-1] = d_eq_I[-1]-coe_c_I[grid.Nw - 1] * F_upper_I

    a_eq_I = coe_a_I[1:grid.Nw]
    b_eq_I = coe_b_I[1:grid.Nw]
    c_eq_I = coe_c_I[1:grid.Nw]
    
    d_eq_II = F_II_ns_old[1:grid.Nw] +grid.dt * model.delta * (a_II_ns_old[1:grid.Nw] - c_II_ns_old[1:grid.Nw])
    d_eq_II[0] = d_eq_II[0]-coe_a_II[1] * F_lower_II
    d_eq_II[-1] = d_eq_II[-1]-coe_c_II[grid.Nw - 1] * F_upper_II

    a_eq_II = coe_a_II[1:grid.Nw]
    b_eq_II = coe_b_II[1:grid.Nw]
    c_eq_II = coe_c_II[1:grid.Nw]
    
    d_eq_III = F_III_ns_old[1:grid.Nw] +grid.dt * model.delta * (a_III_ns_ges_old[1:grid.Nw] - c_III_ns_ges_old[1:grid.Nw])
    d_eq_III[0] = d_eq_III[0]-coe_a_III[1] * F_lower_III
    d_eq_III[-1] = d_eq_III[-1]-coe_c_III[grid.Nw - 1] * F_upper_III

    a_eq_III = coe_a_III[1:grid.Nw]
    b_eq_III = coe_b_III[1:grid.Nw]
    c_eq_III = coe_c_III[1:grid.Nw]
    
    d_eq_IV_p = F_IV_p_ns_old[1:grid.Nw_p] +grid.dt * model.delta * (model.theta*a_IV_p_ns_old[1:grid.Nw_p] - c_IV_p_ns_old[1:grid.Nw_p])
    d_eq_IV_p[0] = d_eq_IV_p[0]-coe_a_IV_p[1] * F_lower_IV_p
    d_eq_IV_p[-1] = d_eq_IV_p[-1]-coe_c_IV_p[grid.Nw_p - 1] * F_upper_IV_p

    a_eq_IV_p = coe_a_IV_p[1:grid.Nw_p]
    b_eq_IV_p = coe_b_IV_p[1:grid.Nw_p]
    c_eq_IV_p = coe_c_IV_p[1:grid.Nw_p]

    # Calculate value of retirement
    value_oo_I = -w_I[1:grid.Nw] ** 2
    value_oo_II = -w_II[1:grid.Nw] ** 2
    value_oo_IV_p = -w_IV_p**2
    
    # Calculate value of replacement
    value_rp = -w_I[1:grid.Nw]**2+D_ns_old

    # Solve equation system
    F_seg_I, ret_seg_I = trisolver(a_eq_I, b_eq_I, c_eq_I, d_eq_I, value_oo_I)
    F_I_ns[1:grid.Nw] = F_seg_I
    ret_I_ns[1:grid.Nw] = ret_seg_I

    F_seg_II, ret_seg_II = trisolver(a_eq_II, b_eq_II, c_eq_II, d_eq_II, value_oo_II)
    F_II_ns[1:grid.Nw] = F_seg_II
    ret_II_ns[1:grid.Nw] = ret_seg_II
    
    F_seg_III, rep_seg_III = trisolver(a_eq_III, b_eq_III, c_eq_III, d_eq_III, value_rp)
    F_III_ns[1:grid.Nw] = F_seg_III
    rep_III_ns[1:grid.Nw] = rep_seg_III
    
    F_seg_IV_p, ret_seg_IV_p = trisolver(a_eq_IV_p, b_eq_IV_p, c_eq_IV_p, d_eq_IV_p, value_oo_IV_p)
    F_IV_p_ns[1:grid.Nw_p] = F_seg_IV_p
    ret_IV_p_ns[1:grid.Nw_p] = ret_seg_IV_p

    # Set values at the boundary
    F_I_ns[grid.Nw] = -w_I[grid.Nw] ** 2
    ret_I_ns[grid.Nw] = 1
    
    F_II_ns[grid.Nw] = -w_II[grid.Nw] ** 2
    ret_II_ns[grid.Nw] = 1
    
    F_III_ns[grid.Nw] = -w_I[grid.Nw] ** 2+D_ns_old
    rep_III_ns[grid.Nw] = 1
    
    F_IV_p_ns[grid.Nw_p] = -w_IV_p[grid.Nw_p] ** 2
    ret_IV_p_ns[grid.Nw_p] = 1

    F_I_ns[0] = 0
    ret_I_ns[0] = 0
    
    F_II_ns[0] = 0
    ret_II_ns[0] = 0
    
    F_III_ns[0] = D_ns_old
    rep_III_ns[0] = 0
    
    F_IV_p_ns[0] = 0
    ret_IV_p_ns[0] = 0
    
    # Calcualte new value of D
    D_ns = np.max(F_III_ns - model.cost)
    pos_replace = np.argmax(F_III_ns - model.cost)
    
    # Calculate lower boundary values for F_IV
    value_bound = [F_IV_p_ns[grid_index_u] - model.K, - w_IV[-1]**2]
    F_lower_IV = 0
    F_upper_IV = np.max(value_bound)
    exe_IV_ns[-1] = np.argmax(value_bound)
    
    # Construct elements for equation system and adjust the size fpr Scenario IV
    d_eq_IV = F_IV_ns_old[1:grid.Nw_p]+grid.dt * model.delta * (a_IV_ns_old[1:grid.Nw_p] - c_IV_ns_old[1:grid.Nw_p])
    d_eq_IV[0] = d_eq_IV[0]-coe_a_IV[1] * F_lower_IV
    d_eq_IV[-1] = d_eq_IV[-1]-coe_c_IV[grid.Nw_p - 1] * F_upper_IV

    a_eq_IV = coe_a_IV[1:grid.Nw_p]
    b_eq_IV = coe_b_IV[1:grid.Nw_p]
    c_eq_IV = coe_c_IV[1:grid.Nw_p]
    
    # Calculate value of outside option
    F_IV_p_stack_ns = np.concatenate([np.zeros((grid_index_l)), F_IV_p_ns[:grid_index_u+1]])
    matrix_oo = np.column_stack([F_IV_p_stack_ns[1:grid.Nw_p] - model.K,-w_IV[1:grid.Nw_p]**2])
    value_oo_IV = np.max(matrix_oo, axis=1)
    exe_prime = np.argmax(matrix_oo, axis=1)+1
    
    # Solve equation system for Scenario IV
    F_seg_IV, exe_seg_IV = trisolver(a_eq_IV, b_eq_IV, c_eq_IV, d_eq_IV, value_oo_IV)
    F_IV_ns[1:grid.Nw_p] = F_seg_IV
    exe_IV_ns[1:grid.Nw_p] = exe_seg_IV
    
    # Set execution values
    exe_IV_ns[0] = 0
    exe_IV_ns[1:grid.Nw_p] = (exe_IV_ns[1:grid.Nw_p] == 1) * exe_prime
    
    # Set values at the boundary
    F_IV_ns[grid.Nw_p] = F_upper_IV
    F_IV_ns[0] = 0;
    
    # Update policies
    c_I_ns, a_I_ns = policy_sen_I_II_p(F_I_ns, ret_I_ns, w_I, 1.0)
    c_II_ns, a_II_ns = policy_sen_I_II_p(F_II_ns, ret_II_ns, w_II, 1.0)
    c_III_ns_ges, a_III_ns_ges, c_III_ns_0, a_III_ns_0, c_III_ns_1, a_III_ns_1 = policy_sen_III(F_III_ns, rep_III_ns, pos_replace, w_I)
    c_IV_p_ns, a_IV_p_ns = policy_sen_I_II_p(F_IV_p_ns, ret_IV_p_ns, w_IV_p, 1.5)
    
    c_p_stack_ns = np.concatenate([np.zeros((grid_index_l)), c_IV_p_ns[:grid_index_u+1]])
    a_p_stack_ns = np.concatenate([np.zeros((grid_index_l)), a_IV_p_ns[:grid_index_u+1]])
    
    c_IV_ns, a_IV_ns = policy_sen_IV(F_IV_ns, exe_IV_ns, a_p_stack_ns, c_p_stack_ns, w_IV)
    
    # Store results
    F_I[:, j] = F_I_ns
    a_I[:, j] = a_I_ns
    c_I[:, j] = c_I_ns
    ret_I[:, j] = ret_I_ns
    
    F_II[:, j] = F_II_ns
    a_II[:, j] = a_II_ns
    c_II[:, j] = c_II_ns
    ret_II[:, j] = ret_II_ns
    
    F_III[:, j] = F_III_ns
    a_III_ges[:, j] = a_III_ns_ges
    a_III_0[:, j] = a_III_ns_0
    a_III_1[:, j] = a_III_ns_1
    c_III_ges[:, j] = c_III_ns_ges
    c_III_0[:, j] = c_III_ns_0
    c_III_1[:, j] = c_III_ns_1
    rep_III[:, j] = rep_III_ns
    D[0, j] = D_ns
    
    F_IV_p[:, j] = F_IV_p_ns
    a_IV_p[:, j] = a_IV_p_ns
    c_IV_p[:, j] = c_IV_p_ns
    ret_IV_p[:, j] = ret_IV_p_ns
    
    F_IV[:, j] = F_IV_ns
    a_IV[:, j] = a_IV_ns
    c_IV[:, j] = c_IV_ns
    exe_IV[:, j] = exe_IV_ns
    
    # Update old values
    F_I_ns_old = F_I_ns.copy()
    c_I_ns_old = c_I_ns.copy()
    a_I_ns_old = a_I_ns.copy()
    ret_I_ns_old = ret_I_ns.copy()
    
    F_II_ns_old = F_II_ns.copy()
    c_II_ns_old = c_II_ns.copy()
    a_II_ns_old = a_II_ns.copy()
    ret_II_ns_old = ret_II_ns.copy()
    
    F_III_ns_old =F_III_ns.copy()
    c_III_ns_ges_old = c_III_ns_ges.copy()
    c_III_ns_0_old = c_III_ns_0.copy()
    c_III_ns_1_old = c_III_ns_1.copy()
    a_III_ns_ges_old = a_III_ns_ges.copy()
    a_III_ns_0_old = a_III_ns_0.copy()
    a_III_ns_1_old = a_III_ns_1.copy()
    D_ns_old = D_ns.copy()
    rep_III_ns_old = rep_III_ns.copy()
    
    F_IV_ns_old = F_IV_ns.copy()
    c_IV_ns_old = c_IV_ns.copy()
    a_IV_ns_old = a_IV_ns.copy()
    exe_IV_ns_old = exe_IV_ns.copy()
    
    F_IV_p_ns_old = F_IV_p_ns.copy()
    c_IV_p_ns_old = c_IV_p_ns.copy()
    a_IV_p_ns_old = a_IV_p_ns.copy()
    ret_IV_p_ns_old = ret_IV_p_ns.copy()

print('Done Value Function Iteration')
print('Plotting Scenario I')

plt.figure()
plt.plot(w_I, F_I[:, 0], color='black', linewidth=1.5)
plt.plot(w_I, F_I_ret, linestyle=':', color=[0.5, 0.5, 0.5], linewidth=1.5)
plt.axhline(0, color='black', linewidth=1.15)
plt.axvline(0.94, linestyle='--', color='black')
plt.text(1.02, -0.75, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Principal's profit $F(w)$", fontsize=11)
plt.text(0.24, 0.12, r'$F(w)$', fontsize=10)
plt.text(0.17, -0.18, r'$F_{0}(w)$', fontsize=10)
plt.ylim([-1.6, 0.2])
plt.xlim([0, 1.2])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_I, c_I[:, 0], color='black', linewidth=1.5)
plt.axvline(0.94, linestyle='-.', color='black')
plt.text(1.02, 2.0, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal consumption $c(w)$", fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([0, 4.0])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_I, a_I[:, 0], color='black', linewidth=1.5)
plt.axvline(0.94, linestyle='-.', color='black')
plt.text(1.02, 0.35, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal effort $a(w)$", fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([0, 0.8])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

ret_I_initial = ret_I[1:, 0]
index_I_ret = np.argmax(ret_I_initial)
w_ret_I = w_I[index_I_ret + 1]

print('Plotting Scenario II')

plt.figure()
plt.plot(w_II, F_II[:, 0], color='black', linewidth=1.5)
plt.plot(w_I, F_I_ret, linestyle=':', color=[0.5, 0.5, 0.5], linewidth=1.5)
plt.axhline(0, color='black', linewidth=1.15)
plt.axvline(0.94, linestyle='-.', color='black')
plt.text(1.02, -0.75, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.axvline(model.w_oo, linestyle='-.', color='black')
plt.text(model.w_oo+0.05, -0.75, r'$\tilde{w}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Principal's profit $F(w)$", fontsize=11)
plt.text(0.24, 0.12, r'$F(w)$', fontsize=11)
plt.text(0.25, -0.26, r'$F_{0}(w)$', fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([-1.6, 0.2])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_II, c_II[:, 0], color='black', linewidth=1.5)
plt.axvline(0.94, linestyle='-.', color='black')
plt.text(1.02, 2.0, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.axvline(model.w_oo, linestyle='--', color='black')
plt.text(model.w_oo+0.05, 2.0, r'$\tilde{w}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal consumption $c(w)$", fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([0, 4.0])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_II, a_II[:, 0], color='black', linewidth=1.5)
plt.axvline(0.94, linestyle='-.', color='black')
plt.text(1.02, 0.3, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.axvline(model.w_oo, linestyle='--', color='black')
plt.text(model.w_oo+0.05, 0.3, r'$\tilde{w}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal effort $a(w)$", fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([0, 0.6])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

ret_II_initial = ret_II[1:, 0]
index_II_ret = np.argmax(ret_II_initial)
w_ret_II = w_II[index_II_ret + 1]

print('Plotting Scenario III')

F_ret_D = F_I_ret + D[0, 0]

plt.figure()
plt.plot(w_I, F_III[:, 0], color='black', linewidth=1.5)
plt.plot(w_I, F_I_ret, linestyle=':', color=[0.5, 0.5, 0.5], linewidth=1.5)
plt.plot(w_I, F_ret_D, linestyle='-.', color=[0.2, 0.2, 0.2], linewidth=1.5)
plt.axhline(0, color='black', linewidth=1.15)
plt.axvline(0.46, linestyle='-.', color='black')
plt.text(0.545, -0.75, r'$w_{rep}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Principal's profit $F(w)$", fontsize=11)
plt.text(0.28, 0.2, r'$F(w)$', fontsize=11)
plt.text(0.01, 0.02, r'$F_{0}(w)+D$', fontsize=11)
plt.text(0.22, -0.22, r'$F_{0}(w)$', fontsize=11)
plt.ylim([-1.6, 0.4])
plt.xlim([0, 1.2])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_I, c_III_ges[:, 0], color='black', linewidth=1.5)
plt.axvline(0.46, linestyle='-.', color='black')
plt.text(0.545, 2.0, r'$w_{rep}$', fontsize=10, ha='center', va='center')
plt.axvline(0.08, linestyle='--', color='black')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal consumption $c(w)$", fontsize=11)
plt.xlim([0, 1.2])
plt.gcf().set_size_inches(3.5, 4)
plt.ylim([0, 4.0])
plt.show()

a_new_agent_x_1 = np.linspace(0.46, 1.2, 1000)
a_new_agent_x_2 = np.linspace(0.0, 0.46, 1000)
a_new_agent_y_1 = np.full(1000, 0.8678)
a_new_agent_y_2 = np.zeros(1000)

plt.figure()
plt.plot(w_I, a_III_0[:, 0], color='black', linewidth=1.5)
plt.plot(a_new_agent_x_2, a_new_agent_y_2, linestyle=':', color=[0.2, 0.2, 0.2], linewidth=1.5)
plt.plot(a_new_agent_x_1, a_new_agent_y_1, linestyle=':', color=[0.2, 0.2, 0.2], linewidth=1.5)
plt.axvline(0.46, linestyle='-.', color='black')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal effort $a(w)$", fontsize=11)
plt.text(0.49, 0.6, r'$w_{rep}$', fontsize=10)
plt.text(0.25, 0.6, r'$a_{old}$', fontsize=11)
plt.text(0.78, 0.9, r'$a_{new}$', fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([0, 1.0])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

taxis = grid.dt*grid.nt

plt.figure()
plt.plot(taxis, D.T, linestyle='-', color='black', linewidth=1.5)
plt.axhline(0, color='black', linewidth=1.15)
plt.xlabel(r'Time $t$', fontsize=11)
plt.ylabel(r'Replacement Profit $D(t)$', fontsize=11)
plt.gcf().set_size_inches(3.5, 4)
plt.xlim([0, 100])
plt.ylim([0, 0.25])
plt.show()

rep_III_initial = rep_III[1:, 0]
index_III_rep = np.argmax(rep_III_initial)
w_rep_III = w_I[index_III_rep + 1]

print('Plotting Scenario IV')

F_IV_cost = F_IV_p[:, 0] - model.K

plt.figure()
plt.plot(w_IV_p, F_IV_p[:, 0], color='black', linewidth=1.5)
plt.plot(w_IV_p, F_ret_IV_p, linestyle=':', color=[0.5, 0.5, 0.5], linewidth=1.5)
plt.axhline(0, color='black', linewidth=1.15)
plt.axvline(1.43, linestyle='-.', color='black')
plt.text(1.3, -2.5, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Principal's profit $F(w)$", fontsize=11)
plt.axvline(model.w_oo_p, linestyle='-.', color='black')
plt.text(model.w_oo_p+0.1, -2.5, r'$\tilde{w}$', fontsize=10, ha='center', va='center')
plt.text(0.45, 0.2, r'$F_{P}(w)$', fontsize=11)
plt.text(0.55, -0.75, r'$F_{0}(w)$', fontsize=11)
plt.xlim([0, 1.8])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_IV_p, c_IV_p[:, 0], color='black', linewidth=1.5)
plt.axvline(1.43, linestyle='-.', color='black')
plt.text(1.57, 1.75, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal consumption $c(w)$", fontsize=11)
plt.axvline(model.w_oo_p, linestyle='-.', color='black')
plt.text(model.w_oo_p+0.1, 1.75, r'$\tilde{w}$', fontsize=10, ha='center', va='center')
plt.xlim([0, 1.8])
plt.gcf().set_size_inches(3.5, 4)
plt.ylim([0, 3.5])
plt.xlim([0, 1.8])
plt.show()

plt.figure()
col = 'black'
lw = 1.5
plt.plot(w_IV_p, a_IV_p[:, 0], color='black', linewidth=1.5)
plt.axvline(1.43, linestyle='-.', color='black')
plt.text(1.45, 0.5, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal effort $a(w)$", fontsize=11)
plt.axvline(model.w_oo_p, linestyle='-.', color='black')
plt.text(model.w_oo_p+0.1, 0.5, r'$\tilde{w}$', fontsize=10, ha='center', va='center')
plt.xlim([0, 1.8])
plt.ylim([0, 0.9])
plt.gcf().set_size_inches(3.5, 4)
plt.ylim([0, 1.0])
plt.xlim([0, 1.8])
plt.show()

plt.figure()
plt.plot(w_IV, F_IV[:, 0], color='black', linewidth=1.5)
plt.plot(w_IV, F_ret_IV, linestyle=':', color=[0.5, 0.5, 0.5], linewidth=1.5)
plt.plot(w_IV_p, F_IV_cost, linestyle='-.', color=[0.5, 0.5, 0.5], linewidth=1.5)
plt.axhline(0, color='black', linewidth=1.15)
plt.axvline(1.05, linestyle='-.', color='black')
plt.text(1.12, -0.75, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.axvline(0.30, linestyle=':', color='black')
plt.text(0.37, -0.75, r'$w_{pr}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Principal's profit $F(w)$", fontsize=11)
plt.text(0.05, 0.08, r'$F(w)$', fontsize=11)
plt.text(0.42, 0.1, r'$F_{P}(w)-K$', fontsize=11)
plt.text(0.42, -0.43, r'$F_{0}(w)$', fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([-1.6, 0.3])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_IV, c_IV[:, 0], color='black', linewidth=1.5)
plt.axvline(1.05, linestyle='-.', color='black')
plt.text(1.13, 0.75, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.axvline(0.30, linestyle='-.', color='black')
plt.text(0.37, 0.75, r'$w_{pr}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal consumption $c(w)$", fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([0.0, 1.5])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

plt.figure()
plt.plot(w_IV, a_IV[:, 0], color='black', linewidth=1.5)
plt.axvline(1.05, linestyle='-.', color='black')
plt.text(1.13, 0.5, r'$w_{ret}$', fontsize=10, ha='center', va='center')
plt.axvline(0.30, linestyle='-.', color='black')
plt.text(0.37, 0.5, r'$w_{pr}$', fontsize=10, ha='center', va='center')
plt.xlabel(r"Agent's continuation value $w$", fontsize=11)
plt.ylabel(r"Optimal effort $a(w)$", fontsize=11)
plt.xlim([0, 1.2])
plt.ylim([0.0, 1.0])
plt.gcf().set_size_inches(3.5, 4)
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Computation time: {elapsed_time:.4f} seconds")