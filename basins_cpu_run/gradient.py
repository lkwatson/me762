NUM_TRIALS = 1000000  #10000000

import cPickle as pickle

import time

import numpy as np
import scipy as sp
from scipy import pi, sin, cos

from scipy import integrate

_theta_rng = 6  # 0+/-val
_omega_rng = 12  # 0+/-val


def f(y, t):
    mu = 1.96  # Friction factor
    m = 1   # Mass
    g = 9.8  # Gravity
    l = 9.8  # Length of string
    d_f = 1.0  # Driving frequency
    tor = 19.6  # Driving toque

    return ( y[1], (-m*g*sin(y[0]) - mu*y[1] + tor*sin(d_f*t))/(m*l) )


def compute_rand_sol():
    np.random.seed()
    theta = _theta_rng*np.random.random(1)[-1] - (_theta_rng/2) # Rand theta
    np.random.seed()
    omega = _omega_rng*np.random.random(1)[-1] - (_omega_rng/2) # Rand omega

    y0 = [theta, omega] # Initial vals
    t_ = np.arange(0, 80, 0.05)

    return integrate.odeint(f, y0, t_), theta, omega, t_

res_s = list()
thetas = list()
omegas = list()
time_line = list()

s_time = time.time()

for i in range(0, NUM_TRIALS):
    if i % 10000 == 0:
        print("Trial {}".format(i))

    res, th, om, t_ = compute_rand_sol()
    res_s.append(res)
    thetas.append(th)
    omegas.append(om)
    time_line.append(t_)

e_time = time.time()

print("Took {} seconds".format(e_time - s_time))

with open('jul14-timeres-odeint-2', 'wb') as fp:
    pickle.dump(res_s, fp)
    pickle.dump(thetas, fp)
    pickle.dump(omegas, fp)
    pickle.dump(time_line, fp)
