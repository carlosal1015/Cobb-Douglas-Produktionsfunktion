"""
Recuperado de:
	http://gappyfacets.com/2015/06/25/python-cobb-douglas-production-function-with-sympy/
"""
from sympy import *
import numpy as np
from matplotlib.pylab import plt

%matplotlib inline
init_printing(use_latex=True)

# Register symbols
var("L K Y A a")

# Cobb-Douglas production function:
Y =  A*(L**a)*K**(1-a)

# Assign number to A and a:
Ys = Y.subs({A:10, a:0.6})

# Plot 3D chart in which K and L are changed 0 to 10
plotting.plot3d(Ys, (K, 0, 10), (L, 0, 10))

# Turn sympy symbols into python function:
Ys_func = lambdify((K, L), Ys, "numpy")

# Make 2D permutation list with K = 0~10 and L = 0~10:
K_n = np.linspace(0, 10, 50)
L_n = np.linspace(0, 10, 50)

result = []

for k in K_n:
	result_j = []
	for l in L_n:
		result_j.append(Ys_func(k, l))
	result.append(result_j)
result = np.array(result)

# Plot 2D heat map:
plt.matshow(result)