import numpy as np
from numpy.random import RandomState
import os

from grad import solve_reuse_hessian, f_iters_reuse_hessian

import cPickle as pickle

"""
Some approximate Newton methods.

q9.31, (519-520), Boyd and Vandenberghe
"""

m,n = (300,150)
prng = RandomState(1) # use instead of numpy.random.seed(). src: http://stackoverflow.com/questions/5836335/consistenly-create-same-random-numpy-array
A = prng.randn(m,n) # each row is a_i^T (n factors), and there are m rows. So A is already in the right shape
x0 = np.zeros((n,1))

x_stars,df_stats = [],[]
file_path = 'data/newtons_method_-_x_stars,df_stats_-_reuse_hessian.pkl'
if os.path.isfile(file_path):
    with open(file_path, 'r') as f:
        x_stars, df_stats = pickle.load(f)
else:
    x_stars, df_stats = solve_reuse_hessian(A, x0)
    with open(file_path, 'w') as f:
        pickle.dump((x_stars, df_stats),f)
f_iters_reuse_hessian(df_stats, n, 'Newton\'s method re-using the Hessian N times',p_star=None, path_template='img/newtons_method_-_reuse_hessian_-_n=1,2,15,30,50_-_rs=1,randn=300,150.png')
## find p_star (it should be somewhere in there)
p_star = 0
for df in df_stats:
    p_star = min(p_star,min(df['f']))
f_iters_reuse_hessian(df_stats, n, 'Newton\'s method re-using the Hessian N times', p_star=p_star,path_template='img/convergence_rate_-_newtons_method_-_reuse_hessian_-_n=1,2,15,30,50_-_rs=1,randn=300,150.png')
