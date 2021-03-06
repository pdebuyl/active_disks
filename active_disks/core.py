import numpy as np
import math

from scipy.integrate import odeint
from scipy.interpolate import BSpline, interp1d, sproot, make_interp_spline
from scipy.spatial.distance import pdist

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

def norm2(x):
    return math.sqrt(x[0]**2+x[1]**2)

def collision_step(pos, vel, f, L):

    N = len(pos)
    radius = 0.5
    L_max = max(*L)*math.sqrt(2)

    t_max = 0
    pairs = []
    for i1, i2 in itertools.combinations(range(N), 2):

        r12 = pos[i1] - pos[i2]
        v1 = vel[i1]
        v2 = vel[i2]
        v12 = v1-v2

        criterion = np.dot(r12, v12)

        if criterion<0:
            # compute the intersection of paths
            w1 = cross_onez(v1)
            w2 = cross_onez(v2)

            t1 = -np.dot(w2, r12)/np.dot(w2, v1)
            t2 = np.dot(w1, r12)/np.dot(w1, v2)

            # keep smallest of the two times where the disk has passed the intersection
            # point plus a full diameter to avoid uselessly large values for t_max
            t_local = min(
                          t1 + 2*radius/norm2(v1),
                          t2 + 2*radius/norm2(v2)
                          )

            # keep the collision only if it occurs inside of the box
            if norm2(v1)*t1 < L_max and norm2(v2)*t2 < L_max:
                pairs.append((i1, i2))
                t_max = max(t_max, t_local)

    for i in range(N):
        t_local = [0, 0]
        for j in [0, 1]:
            if vel[i,j]>0:
                t_local[j] = (L[j]-pos[i,j])/vel[i,j]
            else:
                t_local[j] = -pos[i,j]/vel[i,j]
        # keep only the earliest collision
        t_local = min(t_local)
        t_max = max(t_local, t_max)

    # also check wall collisions for tmax

    if t_max == 0:
        return None, None, None, None

    t = np.linspace(0, t_max, 2000)
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
    wall_dir = -1
    for i in range(N):
        for j in [0, 1]:
            if vel[i,j]>0:
                x_spline = make_interp_spline(t, trajectories[i].x[:,j]+radius-L[j], k=3)
                wall_dir = 1
            else:
                x_spline = make_interp_spline(t, trajectories[i].x[:,j]-radius, k=3)
                wall_dir = 0
            roots = sproot(x_spline)
            if len(roots)>0:
                if roots[0]<wall_collision_time:
                    wall_collision_time = roots[0]
                    wall_collision_idx = i, j, wall_dir

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
                pass #print(roots)
            # check min dist
            if r12_norm[int(roots[0]/dt)] < 2*radius:
                r12_spline = make_interp_spline(t, r12_norm - 2*radius, k=3)
                collision_time = sproot(r12_spline)
                if len(collision_time)>0:
                    collision_time = collision_time[0]
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
    if final_time>50:
        c_type = collision_types.NULL
        final_time = 50
    t = np.linspace(0, final_time, 2000)
    for i in range(N):
        x, v = solve_x_v(f, math.sqrt(vel[i, 0]**2+vel[i, 1]**2), t)
        unit_v = normalize(vel[i])
        x = pos[i] + x[-1]*unit_v
        v = v[-1]*unit_v
        new_x.append(x)
        new_v.append(v)

    return np.array(new_x), np.array(new_v), c_type, final_time, collision_data, t_max

def draw_lines(x, v, t_max):
    import matplotlib.pyplot as plt

    t = np.linspace(0, t_max, 100)
    for i in range(len(x)):
        unit_v = normalize(v[i])
        traj = x[i] + t.reshape((-1,1))*v[i]
        plt.plot(traj[:,0], traj[:,1], label=str(i))

    plt.legend()
    plt.plot(x[:,0], x[:,1], ls='', marker='o')


def collide(x1, v1, x2, v2):
    u12 = normalize(x1 - x2)
    v_com = (v1 + v2) / 2
    kick = np.dot(u12, v2-v1)
    return v1 + kick*u12, v2 - kick*u12


def full_step(pos, vel, N_collisions, f, L, alpha=1, target_kinetic=None, damp_time=None):

    if target_kinetic is not None:
        assert damp_time is not None

    new_x = pos
    new_v = vel
    N = len(pos)
    unique_t = []
    unique_kin = []
    inst_kin = 0
    smooth_kin = []
    store_t = [[0] for i in range(N)]
    store_x = [[pos[i].copy()] for i in range(N)]
    store_v = [[vel[i].copy()] for i in range(N)]
    t = 0
    tot_mom = np.zeros((2, 2))
    n_wall = 0
    n_disk = 0
    t_max_list = []
    k_store = []
    alpha_data = []
    for i in range(N_collisions):
        new_x, new_v, c_type, c_time, c_data, t_max = collision_step(new_x, new_v, f, L)
        t += c_time
        t_max_list.append(t_max)
        unique_t.append(t)
        unique_kin.append(np.sum(new_v**2)/(2*N))
        loop_kin = np.sum(new_v**2)/(2*N)
        inst_kin = (inst_kin*i + loop_kin) / (i+1)
        smooth_kin.append(inst_kin)
        k_store.append( (loop_kin + sum(k_store[-9:]))/10 )
        # update alpha to aim for the target kinetic temperature
        if target_kinetic is not None and i>20 and c_type != collision_types.NULL:
            alpha = alpha + (target_kinetic-k_store[-1])/damp_time
            alpha = min(alpha, 1)
            alpha = max(0.01, alpha)

        if target_kinetic is not None and i%100==0:
            print(alpha)
        alpha_data.append(alpha)
        if c_data == (-1, -1):
            print('No more collisions')
            break
        if c_type == collision_types.DISKS:
            n_disk = n_disk + 1
            i1, i2 = c_data
            new_v1, new_v2 = collide(new_x[c_data[0]], new_v[c_data[0]], new_x[c_data[1]], new_v[c_data[1]])
            new_v[c_data[0]] = new_v1
            new_v[c_data[1]] = new_v2
            store_t[i1].append(t)
            store_t[i2].append(t)
            store_x[i1].append(new_x[i1])
            store_x[i2].append(new_x[i2])
            store_v[i1].append(new_v[i1])
            store_v[i2].append(new_v[i2])
        elif c_type == collision_types.WALL:
            n_wall = n_wall + 1
            v = new_v[c_data[0],c_data[1]]
            tot_mom[c_data[1], c_data[2]] += v
            new_v[c_data[0],c_data[1]] = -alpha*v
            store_t[c_data[0]].append(t)
            store_x[c_data[0]].append(new_x[c_data[0]])
            store_v[c_data[0]].append(new_v[c_data[0]])

    store_x = [np.array(store_x[i]) for i in range(N)]
    store_v = [np.array(store_v[i]) for i in range(N)]
    print(n_wall, 'collisions with the walls')
    print(n_disk, 'collisions between disks')

    return np.array(unique_t), np.array(unique_kin), store_t, store_x, store_v, t, tot_mom, t_max_list, np.array(smooth_kin), alpha_data

def draw_t_x(tt, xx):
    import matplotlib.pyplot as plt
    # make a set of unique collision times
    a = set(tt[0])
    for t_elem in tt[1:]:
        a.update(set(t_elem))

    ax1 = plt.subplot(211)
    [plt.plot(tt[i], xx[i][:,0]) for i in range(9)]
    [plt.axvline(t) for t in a]
    ax2 = plt.subplot(212, sharex=ax1)
    [plt.plot(tt[i], xx[i][:,1]) for i in range(9)]
    [plt.axvline(t) for t in a]


def generate_ic(N, L, radius):
    L = np.array(L).reshape((1, 2))
    while True:
        x = radius + np.random.uniform(size=(N, 2)) * (L - 2*radius)
        if np.min(pdist(x)) > 2*radius:
            break
    v = np.random.normal(size=(N, 2))
    return x, v
