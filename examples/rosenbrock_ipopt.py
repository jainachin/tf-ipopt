"""
Copyright 2019 Achin Jain (achinj@seas.upenn.edu)

@problem: solve HS071 using Tensorflow and IpoptOptimizer

  minimize        x^2 + 100z^2
  subject to      z + (1-x)^2 - y = 0

"""

import os
import tensorflow as tf
import numpy as np

# objective funciton for the optimization f(x)
with tf.name_scope('objective'):
	x0 = np.array([2.5])
	y0 = np.array([3.0])
	z0 = np.array([0.75])
	x = tf.Variable(x0, name='x', dtype=tf.float32)
	y = tf.Variable(y0, name='y', dtype=tf.float32)
	z = tf.Variable(z0, name='z', dtype=tf.float32)
	f = x**2 + 100*(z**2)

# all inequalities are expressed as g(x)>=0 and equalities as h(x)=0
with tf.name_scope('constraints'):
	h = z + (1-x)**2 - y

# set optimizer
optimizer = tf.contrib.opt.IpoptOptimizerInterface(loss=f,
													var_list=[x, y, z],
													inequalities=[],
													equalities=[h],
													var_to_bounds={},
												   	options={
												   	"print_level": 5, 
												 	"max_iter": 300,
												 	"linear_solver": "mumps",
												 	"log_level": 0,
												 	"hessian_approximation": "limited-memory"
												 	}
												 	)

# initialize
init = tf.global_variables_initializer()
    
with tf.Session() as session:
    session.run(init)
    optimizer.minimize(session, feed_dict={})
    print('fun: ', session.run(f, feed_dict={}))
    print('x: ', session.run(x))