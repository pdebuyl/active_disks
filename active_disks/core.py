import numpy as np
import math

from scipy.integrate import odeint
from scipy.interpolate import BSpline, interp1d, sproot, make_interp_spline

import itertools
from collections import namedtuple
from enum import Enum

trajectory = namedtuple('trajectory', ['x', 'v'])

collision_types = Enum('collision', 'NULL WALL DISKS')

def cross_onez(x):
    return np.cross([0, 0, 1], x)[:2]

def normalize(x):
    x = np.copy(x)
    return x/np.sqrt(np.dot(x, x))

def solve_x_v(f, v0, t):
    def local_f(x, t):
        return x[1], f(x[1])
    x0v0 = (0, v0)
    trajectory = odeint(local_f, x0v0, t)
    return trajectory[:,0].reshape((-1,1)), trajectory[:,1].reshape((-1,1))


def pair_time(r, v, i1, i2, f):
    r12 = r[i1] - r[i2]
    v1 = v[i1]
    unit_v1 = normalize(v1)
    v2 = v[i2]
    unit_v2 = normalize(v2)
    v12 = v1-v2

    radius = 0.5

    criterion = np.dot(r12, v12)

    if criterion >= 0:
        return 0, 0, -1

    w1 = cross_onez(v1)
    w2 = cross_onez(v2)

    t1 = -np.dot(w2, r12)/np.dot(w2, v1)
    t2 = np.dot(w1, r12)/np.dot(w1, v2)

    print(t1, t2)

    t_max = max(t1, t2)
    t = np.linspace(0, t_max, 2000)
    dt = t[1]-t[0]

    # solve the ode for the scalar velocities
    x, v = solve_x_v(f, math.sqrt(v1[0]**2+v1[1]**2), t)

    x1 = r[i1] + x*unit_v1
    v1 = v*unit_v1

    x, v = solve_x_v(f, math.sqrt(v2[0]**2+v2[1]**2), t)

    x2 = r[i2] + x*unit_v2
    v2 = v*unit_v2

    r12 = x1 - x2
    r12_norm = np.sqrt(np.sum(r12**2, axis=1))
    v12 = v1 - v2

    r12_dot = np.sum(r12*v12, axis=1) / r12_norm

    r12_dot_spline = make_interp_spline(t, r12_dot, k=3)
    roots = sproot(r12_dot_spline)
    collision_time = 0
    if len(roots)>0:
        if len(roots)!=1:
            raise Exception('More than one root')
        # check min dist
        if r12_norm[int(roots[0]/dt)] < 2*radius:
            r12_spline = make_interp_spline(t, r12_norm - 2*radius, k=3)
            collision_time = sproot(r12_spline)[0]
            print(sproot(r12_spline))
        else:
            # no collision
            pass

    return t, x1, v1, x2, v2, r12_norm, collision_time


def collision_step(pos, vel, f, L):

    N = len(pos)
    radius = 0.5

    t_max = 0
    pairs = []
    for i1, i2 in itertools.combinations(range(N), 2):

        r12 = pos[i1] - pos[i2]
        v1 = vel[i1]
        v2 = vel[i2]
        unit_v2 = normalize(v2)
        v12 = v1-v2


        criterion = np.dot(r12, v12)

        if criterion<0:
            pairs.append((i1, i2))

            w1 = cross_onez(v1)
            w2 = cross_onez(v2)

            t1 = -np.dot(w2, r12)/np.dot(w2, v1)
            t2 = np.dot(w1, r12)/np.dot(w1, v2)

            t_max = max(t_max, t1, t2)

    for i in range(N):
        for j in [0, 1]:
            if vel[i,j]>0:
                wall_t_max = (L[j]-pos[i,j])/vel[i,j]
            else:
                wall_t_max = pos[i,j]/vel[i,j]
            t_max = max(wall_t_max, t_max)

    # also check wall collisions for tmax

    if t_max == 0:
        return None, None, None, None

    t = np.linspace(0, t_max, 3000)
    dt = t[1]-t[0]

    trajectories = []
    for i in range(N):
        x, v = solve_x_v(f, math.sqrt(vel[i, 0]**2+vel[i, 1]**2), t)
        unit_v = normalize(vel[i])
        x = pos[i] + x*unit_v
        v = v*unit_v
        trajectories.append(trajectory(x=x, v=v))

    wall_collision_time = t_max
    wall_collision_idx = -1, -1
    for i in range(N):
        for j in [0, 1]:
            if vel[i,j]>0:
                x_spline = make_interp_spline(t, trajectories[i].x[:,j]+radius-L[0], k=3)
            else:
                x_spline = make_interp_spline(t, trajectories[i].x[:,j]-radius, k=3)
            roots = sproot(x_spline)
            if len(roots)>0:
                if roots[0]<wall_collision_time:
                    wall_collision_time = roots[0]
                    wall_collision_idx = i, j

    collisions = []
    c_time = t_max
    c_pair = (-1, -1)
    for i1, i2 in pairs:
        r12 = trajectories[i1].x - trajectories[i2].x
        r12_norm = np.sqrt(np.sum(r12**2, axis=1))
        v12 = trajectories[i1].v - trajectories[i2].v

        r12_dot = np.sum(r12*v12, axis=1) / r12_norm

        r12_dot_spline = make_interp_spline(t, r12_dot, k=3)
        roots = sproot(r12_dot_spline)
        collision_time = 0
        if len(roots)>0:
            if len(roots)!=1:
                raise Exception('More than one root')
            # check min dist
            if r12_norm[int(roots[0]/dt)] < 2*radius:
                r12_spline = make_interp_spline(t, r12_norm - 2*radius, k=3)
                collision_time = sproot(r12_spline)[0]
                collisions.append((collision_time, i1, i2))
                if collision_time < c_time:
                    c_time = collision_time
                    c_pair = (i1, i2)

    c_type = collision_types.DISKS if c_time < wall_collision_time else collision_types.WALL

    if c_type == collision_types.DISKS:
        final_time = c_time
        collision_data = c_pair
    elif c_type == collision_types.WALL:
        final_time = wall_collision_time
        collision_data = wall_collision_idx

    new_x = []
    new_v = []
    t = np.linspace(0, final_time, 2000)
    for i in range(N):
        x, v = solve_x_v(f, math.sqrt(vel[i, 0]**2+vel[i, 1]**2), t)
        unit_v = normalize(vel[i])
        x = pos[i] + x[-1]*unit_v
        v = v[-1]*unit_v
        new_x.append(x)
        new_v.append(v)

    return np.array(new_x), np.array(new_v), c_type, final_time, collision_data

def draw_lines(x, v, t_max):
    import matplotlib.pyplot as plt

    t = np.linspace(0, t_max, 100)
    for i in range(len(x)):
        unit_v = normalize(v[i])
        traj = x[i] + t.reshape((-1,1))*unit_v
        plt.plot(traj[:,0], traj[:,1], label=str(i))

    plt.legend()
    plt.plot(x[:,0], x[:,1], ls='', marker='o')


def collide(x1, v1, x2, v2):
    u12 = normalize(x1 - x2)
    v_com = (v1 + v2) / 2
    kick = np.dot(u12, v2-v1)
    return v1 + kick*u12, v2 - kick*u12


def full_step(pos, vel, f, L):
    all_x = []
    all_v = []
    new_x = pos
    new_v = vel
    all_x.append(new_x)
    all_v.append(new_v)
    for i in range(1000):
        new_x, new_v, c_type, c_time, c_data = collision_step(new_x, new_v, f, L)
        print(c_type, c_time)
        if c_data == (-1, -1):
            print('No more collisions')
            break
        if c_type == collision_types.DISKS:
            print("c_data", c_data)
            print("before collide")
            print(new_x[c_data[0]], new_v[c_data[0]], new_x[c_data[1]], new_v[c_data[1]])
            new_v1, new_v2 = collide(new_x[c_data[0]], new_v[c_data[0]], new_x[c_data[1]], new_v[c_data[1]])
            print("after collide")
            new_v[c_data[0]] = new_v1
            new_v[c_data[1]] = new_v2
        elif c_type == collision_types.WALL:
            print(new_x[c_data[0]], new_v[c_data[0]])
            new_v[c_data[0],c_data[1]] *= -1
        all_x.append(new_x)
        all_v.append(new_v)

    return np.array(all_x), np.array(all_v)
