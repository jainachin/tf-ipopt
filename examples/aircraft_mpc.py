
"""
Copyright 2019 Achin Jain (achinj@seas.upenn.edu)

MPC design for aircraft. Example taken from MPC Course at UPenn.
Modeling is done in tensorflow, optimization is solved using Ipopt.

Linearized discrete-time model (at altitude of 5000m and a speed of 128.2 m/sec):
See A, B, C matrices below
Input: elevator angle
States: x1: angle of attack, x2: pitch angle, x3: pitch rate, x4: altitude
Outputs: pitch angle and altitude
Constraints: elevator angle ±0.262rad (±15◦), pitch angle ±0.349 (±39◦)

"""

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MPC params
Q = np.identity(4, dtype=np.float32)
R = 10*np.array([[1]], dtype=np.float32)
horizon = 10

A = np.array([[ 0.6136,         0,    0.1570,         0],
             [ -0.1280,    1.0000,    0.1904,         0],
             [ -0.8697,         0,    0.5248,         0],
             [-27.5636,   32.0500,    0.4087,    1.0000]], dtype=np.float32)

B = np.array([ -0.4539,   -0.4436,   -3.1989,    0.6068], dtype=np.float32).reshape(-1,1)

C = np.array([[      0,         1,         0,         0], 
              [      0,         0,         0,         1]], dtype=np.float32)

# aircraft model
def model(x_k, u_k):
  
    x_k = tf.matmul(A, x_k) + tf.matmul(B, u_k)
    return x_k

def plot_results(xlog, ulog):

    t = [0.25*x for x in range(xlog.shape[1])]

    plt.figure()
    plt.subplot(311)
    plt.plot(t, xlog[3,:], label="altitude", color="b")
    plt.ylabel('altitude')

    plt.subplot(312)
    plt.plot(t, xlog[1,:], label="pitch angle", color="r")
    plt.ylabel('pitch angle')

    plt.subplot(313)
    plt.plot(t, ulog[0,:], label="elevator angle", color="g")
    plt.ylabel('elevator angle')

    plt.show()

# objective function for the optimization
with tf.name_scope('cost'):

    x0 = tf.placeholder(shape=(4,1), name="xinit", dtype=tf.float32)
    uinit = np.zeros([1, horizon])
    inputs = tf.Variable(uinit, 'u', dtype=tf.float32)
    xinit = np.zeros([4, horizon])
    states = tf.Variable(xinit, 'x', dtype=tf.float32)

    f = 0
    for idh in range(horizon):
        u = tf.reshape(inputs[:,idh],[-1,1])
        x = tf.reshape(states[:,idh],[-1,1])
        f = f + tf.matmul(tf.transpose(x), tf.matmul(Q,x)) + tf.matmul(tf.transpose(u), tf.matmul(R,u))

# all inequalities are expressed as g(x)>=0 and h(x)=0
with tf.name_scope('constraints'):
    h = []
    x = x0
    for idh in range(horizon):
        u = tf.reshape(inputs[:,idh],[-1,1])
        cons = tf.reshape(states[:,idh], [-1,1]) - model(x, u)
        for idc in range(4):
            h.append(cons[idc])
        x = tf.reshape(states[:,idh],[-1,1])

    g = []

# set optimizer
optimizer = tf.contrib.opt.IpoptOptimizerInterface(loss=f,
                                                 var_list=[inputs, states],
                                                 inequalities=g,
                                                 equalities=h,
                                                 var_to_bounds={inputs:(-0.262,0.262),
                                                                states: ([[-np.infty],[-0.349],[-np.infty],[-np.infty]], 
                                                                         [[ np.infty],[ 0.349],[ np.infty],[ np.infty]])},
                                                 options={
                                                 "print_level": 2, 
                                                 "max_iter": 100,
                                                 "linear_solver": "mumps",
                                                 "log_level": 0,
                                                 }
                                                 )

def run_controller(n_steps):

    with tf.Session() as session:

        # initialize all variables
        session.run(inputs.initializer)
        session.run(states.initializer)

        xinit = np.array([0., 0., 0., 10.], dtype=np.float32).reshape(4,1)

        # main sim loop
        xlog = np.zeros([4,n_steps])
        ulog = np.zeros([1,n_steps])
        for step in range(n_steps):

            start = time.time()
            # compute mean, variance and their gradients
            feed_dict = {x0: xinit}
            optimizer.minimize(session, feed_dict=feed_dict)
            fopt, uopt = session.run([f, inputs], feed_dict=feed_dict)
            end = time.time()
            print("Time required by IPOPT: {:.2f}".format(end-start))
            
            # store trajectory for plots
            xlog[:,step] = xinit.reshape(-1,)
            ulog[:,step] = uopt[0,0].reshape(-1,)
            
            # simulate model
            xinit = model(xinit, uopt[0,0].reshape(-1,1)).eval()

            print('step: {}, u: {:.2f}'.format(step, uopt[0,0]))
            print('\n')

    return xlog, ulog


if __name__ == "__main__":

    xlog, ulog = run_controller(n_steps=20)
    plot_results(xlog, ulog)
