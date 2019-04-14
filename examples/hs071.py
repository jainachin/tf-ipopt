"""
Copyright 2019 Achin Jain (achinj@seas.upenn.edu)

@problem: solve HS071 using Tensorflow and IpoptOptimizer
https://www.coin-or.org/Ipopt/documentation/node20.html

  minimize        x1 * x4 * (x1 + x2 + x3) + x3
  subject to      x1 * x2 * x3 * x4 >=25
  				  x1^2 + x2^2 + x3^2 + x4^2 = 40
  				  1 <= x1, x2, x3, x4 <= 5
"""

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np

# objective funciton for the optimization f(x)
with tf.name_scope('objective'):
	x1 = tf.Variable(1.0, name='x', dtype=tf.float32)
	x2 = tf.Variable(5.0, name='x', dtype=tf.float32)
	x3 = tf.Variable(5.0, name='x', dtype=tf.float32)
	x4 = tf.Variable(1.0, name='x', dtype=tf.float32)
	f = x1 * x4 * (x1 + x2 + x3) + x3

# all inequalities are expressed as g(x)>=0 and equalities as h(x)=0
with tf.name_scope('constraints'):
	g = (x1 * x2 * x3 * x4 - 25)
	h = x1*x1 + x2*x2 + x3*x3 + x4*x4 - 40

# set optimizer
optimizer = tf.contrib.opt.IpoptOptimizerInterface(loss=f, 
													inequalities=[g],
													equalities=[h],
													var_to_bounds={x1:(1, 5), x2:(1,5), x3:(1, 5), x4:(1,5)},
												   	options={
												   	"print_level": 5, 
												 	# "max_iter": 300,
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
    print('x: ', session.run([x1, x2, x3, x4]))