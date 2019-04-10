#
# Copyright 2019 Achin Jain (achinj@seas.upenn.edu)
# IPOPT for TensorFlow
# 

class IpoptOptimizerInterface(ExternalOptimizerInterface):
  """Wrapper allowing `pyipopt.solve` to operate a `tf.Session`.

  Specify the problem in the format below
  minimize        f(x)
  subject to      g(x)>=0       // inequality
                  h(x)=0        // equality
                  x_L<=x<=x_U   // bounds (can also be written as inequality)
  
  Example 1:

  minimize        x^2 + y^2
  subject to      x>=1, y>=1
  
  ```python

  import os
  import tensorflow as tf
  import numpy as np

  with tf.name_scope('objective'):
    x = tf.Variable(tf.ones([2,]), name='x', dtype=tf.float32)
    c = tf.placeholder(tf.float32, shape=(1,), name='constant')
    f = tf.reduce_sum(tf.square(x)) + c

  optimizer = tf.contrib.opt.IpoptOptimizerInterface(loss=f,
                          var_to_bounds={x: (1,10)}, 
                          options={'max_iter': 100})

  init = tf.global_variables_initializer()
  
  with tf.Session() as session:
    session.run(init)
    optimizer.minimize(session, feed_dict={c: np.array([0])})
    print('fun: ', session.run(f, feed_dict={c: np.array([0])}))
    print('x: ', session.run(x))

  # The value of x should now be [1., 1.].
  ```

  Example 2: hs071 from IPOPT with both equality and inequality constraints:
    https://www.coin-or.org/Ipopt/documentation/node23.html#SECTION00053100000000000000

  ```python
  
  # objective funciton for the optimization f(x)
  with tf.name_scope('objective'):
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    x = tf.Variable(x0, name='x', dtype=tf.float32)
    c = tf.placeholder(tf.float32, shape=(1,), name='constant')
    f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2] + c

  # all inequalities are expressed as g(x)>=0 and equalities as h(x)=0
  with tf.name_scope('constraints'):
  g = (x[0] * x[1] * x[2] * x[3] - 25)
  h = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] - 40

  # set optimizer
  optimizer = tf.contrib.opt.IpoptOptimizerInterface(loss=f, 
                          inequalities=[g],
                          equalities=[h],
                          var_to_bounds={x:(1, 5)},
                            options={
                          "print_level": 0, 
                          "max_iter": 100,
                          "linear_solver": "mumps",
                          "log_level": 0}
                          )

  # initialize
  init = tf.global_variables_initializer()
  
  with tf.Session() as session:
    session.run(init)
    optimizer.minimize(session, feed_dict={c: np.array([0])})
    print('fun: ', session.run(f, feed_dict={c: np.array([0])}))
    print('x: ', session.run(x))

  # The value of x should now be [1.0, 4.74, 3.82, 1.37].
  ```
  """

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):

    # initial value should be float64
    initial_val = np.array(initial_val, dtype=np.float_)
    
    # objective function
    def eval_f(x, user_data = None):
      loss, _ = loss_grad_func(x)
      return np.array(loss, dtype=np.float_)

    # gradient of objective function
    def eval_grad_f(x, user_data = None):
      _, grad_f = loss_grad_func(x)
      return np.array(grad_f, dtype=np.float_)

    # gradient function (first inequalities then equalities)
    def eval_g(x, user_data = None):
      inequalities = [inequality_funcs[i](x) for i in range(nineqcon)]
      equalities = [equality_funcs[i](x) for i in range(neqcon)]
      return np.array(inequalities + equalities, dtype=np.float_).reshape(ncon,)

    # hessian of the lagrangian (first inequalities then equalities)
    def eval_h(x, lagrange, obj_factor, flag, user_data = None):
      rows, cols = np.tril_indices(nvar)
      if flag:
        return (np.array(rows, dtype=np.int_), np.array(cols, dtype=np.int_))
      else:
        loss = [loss_hessian_func(x)]
        inequalities = [inequality_hessian_funcs[i](x) for i in range(nineqcon)]
        equalities = [equality_hessian_funcs[i](x) for i in range(neqcon)]
        values = np.zeros([nvar, nvar])
        values += obj_factor*loss[0][0]
        for idc in range(nineqcon):
          values += lagrange[idc]*inequalities[idc][0]
        for idc in range(neqcon):
          values += lagrange[idc+nineqcon]*equalities[idc][0]
        return np.array(values.reshape(nvar,nvar)[rows,cols], dtype=np.float_)

    # jacobian for gradient (first inequalities the equalities)
    def eval_jac_g(x, flag, user_data = None):
      rows, cols = np.indices((ncon,nvar))
      if flag:
        return (np.array(rows.reshape(-1,1), dtype=np.int_), np.array(cols.reshape(-1,1), dtype=np.int_))
      else:
        inequalities = [inequality_grad_funcs[i](x) for i in range(nineqcon)]
        equalities = [equality_grad_funcs[i](x) for i in range(neqcon)]
        values = np.empty([ncon,nvar])
        for idc in range(nineqcon):
          values[idc,:] = inequalities[idc][0]
        for idc in range(neqcon):
          values[idc+nineqcon,:] = equalities[idc][0]
        return np.array(values.reshape(ncon*nvar,), dtype=np.float_)

    # box constraints on the variables
    nvar = int(np.sum([np.prod(self._vars[i].get_shape().as_list()) for i in range(len(self._vars))]))
    if self._packed_bounds is None:
      x_L = -np.ones((nvar), dtype=np.float_) * np.inf
      x_U = np.ones((nvar), dtype=np.float_) * np.inf
    else:
      x_L, x_U = zip(*self._packed_bounds)
      x_L = np.array(x_L, dtype=np.float_)
      x_U = np.array(x_U, dtype=np.float_)
    
    # inequality constraints as g(x)>=0 and equality constraints as h(x)=0
    nineqcon = len(self._inequalities)
    neqcon = len(self._equalities)
    ncon = nineqcon + neqcon
    g_L_ineq = np.zeros((nineqcon), dtype=np.float_)
    g_U_ineq = np.ones((nineqcon), dtype=np.float_) * 2.0*pow(10.0, 19)
    g_L_eq = np.zeros((neqcon), dtype=np.float_)
    g_U_eq = np.zeros((neqcon), dtype=np.float_)
    g_L = np.concatenate((g_L_ineq, g_L_eq), axis=0)
    g_U = np.concatenate((g_U_ineq, g_U_eq), axis=0)
    nnzj = nvar*ncon
    nnzh = int(nvar*(nvar+1)/2)

    minimize_args = [nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g]
    
    # create nlp in ipopt
    import pyipopt
    
    # log_level decides if logs from pyipopt are desired -- these are logs on 
    # top of what is returned from ipopt set by "print_level"; see below
    if "log_level" in optimizer_kwargs["options"]:
      pyipopt.set_loglevel(optimizer_kwargs["options"]["log_level"])

    nlp = pyipopt.create(*minimize_args)
      
    # check https://www.coin-or.org/Ipopt/documentation/node40.html 
    # for more options and default settings
    # default print_level=5
    # default max_iter=3000
    # default tol=1e-8

    for optvar in optimizer_kwargs["options"]:
      if optvar is "log_level":
         print
      elif type(optimizer_kwargs["options"][optvar]) is np.str:
         nlp.str_option(optvar, optimizer_kwargs["options"][optvar])
      elif type(optimizer_kwargs["options"][optvar]) is np.int:
         nlp.int_option(optvar, optimizer_kwargs["options"][optvar])
      else:
         nlp.num_option(optvar, optimizer_kwargs["options"][optvar])     

    result_x, zl, zu, constraint_multipliers, result_f, status = nlp.solve(initial_val)
    nlp.close()
    
    message_lines = [
        'Optimization terminated with:',
        '  Message: %s',
        '  Objective function value: %f',
    ]
    message_args = [status, result_f]
    logging.info('\n'.join(message_lines), *message_args)
    print("Optimization terminated with message: {}".format(status))
    return result_x    