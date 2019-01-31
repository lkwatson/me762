NUM_TRIALS = 10000000

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
    t_ = np.arange(0, 100, 0.01)

    return integrate.odeint(f, y0, t_), theta, omega, t_

thetas_cw_ = list()
omegas_cw_ = list()

thetas_ccw_ = list()
omegas_ccw_ = list()

s_time = time.time()

for i in range(0, NUM_TRIALS):
    if i % 100000 == 0:
        print("Trial {}".format(i))

    res, th, om, t_ = compute_rand_sol()
    #plt.plot(t_,res[:,0]),plt.xlabel('Angle'),plt.ylabel('')
    if res[-1,0] > 0:
        thetas_cw_.append(th)
        omegas_cw_.append(om)
    else:
        thetas_ccw_.append(th)
        omegas_ccw_.append(om)

e_time = time.time()

print("Took {} seconds".format(e_time-s_time))

with open('jul11-run1','wb') as fp:
    pickle.dump(thetas_cw_, fp)
    pickle.dump(omegas_cw_, fp)
    pickle.dump(thetas_ccw_, fp)
    pickle.dump(omegas_ccw_, fp)
