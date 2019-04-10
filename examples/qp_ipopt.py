"""
Copyright 2019 Achin Jain (achinj@seas.upenn.edu)

@problem: solve QP using Tensorflow and IpoptOptimizer

  minimize        x^2 + y^2
  subject to      x>=1, y>=2

"""

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np

with tf.name_scope('objective'):
    x = tf.Variable(np.ones([2,]), name='x', dtype=tf.float32)
    c = tf.placeholder(tf.float32, shape=(1,), name='constant')
    f = tf.reduce_sum(tf.square(x)) + c

# set optimizer
optimizer = tf.contrib.opt.IpoptOptimizerInterface(loss=f,
													var_to_bounds={x: ([1,2], np.infty)}, 
													options={'max_iter': 300,
                                   'linear_solver': 'mumps'}
													)
# initialize
init = tf.global_variables_initializer()
    
with tf.Session() as session:
    session.run(init)
    optimizer.minimize(session, feed_dict={c: np.array([0])})
    print('fun: ', session.run(f, feed_dict={c: np.array([0])}))
    print('x: ', session.run(x))