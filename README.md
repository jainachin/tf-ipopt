## Use Interior Point OPTimizer (IPOPT) with TensorFlow


Follow the steps to install. It is assumed that
* tensorflow
* [ipopt](https://www.coin-or.org/Ipopt/)
* [pyipopt](https://github.com/xuy/pyipopt)

are already installed. Details on how to install ipopt and pyipopt will be added later.

#### Step 1
We define a new class `IpoptOptimizerInterface` that can be used very much like `ScipyOptimizerInterface`. Make the following two changes in the `external_optimizer.py` in tensorflow package located at `tensorflow/contrib/opt/python/training/`.  

Allow import of the new `IpoptOptimizerInterface` class.
```
__all__ = ['ExternalOptimizerInterface', 'ScipyOptimizerInterface', 'IpoptOptimizerInterface']
```

#### Step 2
Second, **copy** the class definition from the file `ipopt_optimizer.py` to `external_optimizer.py`.

#### Step 3
Modify `_allowed_symbols` to add `IpoptOptimizerInterface` in `__init__.py` located at `tensorflow/contrib/opt/`

```
_allowed_symbols = [
    'PowerSignOptimizer',
    'AddSignOptimizer'
    'DelayCompensatedGradientDescentOptimizer',
    'DropStaleGradientOptimizer',
    'ExternalOptimizerInterface',
    'LazyAdamOptimizer',
    'NadamOptimizer',
    'MovingAverageOptimizer',
    'ScipyOptimizerInterface',
    'IpoptOptimizerInterface',
    'VariableClippingOptimizer',
    'MultitaskOptimizerWrapper',
    'clip_gradients_by_global_norm',
    'ElasticAverageOptimizer',
    'ElasticAverageCustomGetter'
]
```

That's it!