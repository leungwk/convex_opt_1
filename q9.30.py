from numpy.random import RandomState
import numpy as np

from grad import solve, gradient_method, newtons_method, f_iters, f_ps
import cPickle as pickle
import os

"""
Gradient and Newton methods.

q9.30, (519), Boyd and Vandenberghe
"""

m,n = (300,150)
prng = RandomState(1) # use instead of numpy.random.seed(). src: http://stackoverflow.com/questions/5836335/consistenly-create-same-random-numpy-array
A = prng.randn(m,n) # each row is a_i^T (n factors), and there are m rows. So A is already in the right shape
x0 = np.zeros((n,1))

x_stars,df_stats = [],[]
file_path = 'data/gradient_method_-_x_stars,df_stats.pkl'
if os.path.isfile(file_path):
    with open(file_path, 'r') as f:
        x_stars, df_stats = pickle.load(f)
else:
    x_stars, df_stats = solve(A, x0, gradient_method)
    with open(file_path, 'w') as f:
        pickle.dump((x_stars, df_stats),f)
f_iters(df_stats, r'Gradient descent for several $\alpha,\beta$')
f_ps(df_stats, A, x_stars, r'Convergence rate of gradient descent')


x_stars,df_stats = [],[]
file_path = 'data/newtons_method_-_x_stars,df_stats.pkl'
if os.path.isfile(file_path):
    with open(file_path, 'r') as f:
        x_stars, df_stats = pickle.load(f)
else:
    x_stars, df_stats = solve(A, x0, newtons_method)
    with open(file_path, 'w') as f:
        pickle.dump((x_stars, df_stats),f)
f_iters(df_stats, r"Newton's method for several $\alpha,\beta$",'img/newtons_method_-_several_alpha,beta_-_rs=1,randn=300,150.png')
f_ps(df_stats, A, x_stars, r"Convergence rate of Newton's method",'img/convergence_rate_-_newtons_method_-_several_alpha,beta_-_rs=1,randn=300,150.png')
