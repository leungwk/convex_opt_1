import numpy as np
from numpy.random import RandomState
import os
import pandas as pd

from lp import run_newton, _plot_newton_convergence, _f, _grad

image_dir = 'img/'

## generate test data
m,n = (100,500)
prng = RandomState(1)
A = prng.randn(m,n)

## check that A is full rank
assert np.linalg.matrix_rank(A) == m
## src: http://stackoverflow.com/a/3356123
u, s, v = np.linalg.svd(A)
assert np.sum(s > 1e-10) == m

## subtracting the mean reduces rank by 1 ...
# mu=np.mean(A,axis=0)
# u,s,v=np.linalg.svd(A-mu)
# np.sum(s > 1e-10) == 99
#
# from sklearn.decomposition import PCA
# sum(PCA().fit(A).explained_variance_ > 1e-10) == 99
#
# np.linalg.matrix_rank(A-mu) == 99

x0 = np.abs(prng.randn(n,1)) # make positive
b = np.dot(A,x0)
c = prng.rand(n,1)



## part 1
df_newton_stats, df_op = run_newton(A, b, c, x0)
image_path = image_dir +'lp_part_1_-_newton_convergence.png'
_plot_newton_convergence(df_newton_stats, df_op, image_path)




## compare against cvxpy
import cvxopt
import cvxpy

x = cvxpy.Variable(n,name='x')
A1 = cvxopt.matrix(A)
c1 = cvxopt.matrix(c)
ones = cvxopt.matrix(np.ones(n))
b1 = cvxpy.Variable(b.shape[0],name='b')
objective = cvxpy.Minimize(
    c1.T*x -ones.T*cvxpy.log(x)
)
constraints = [A1*x == b1]
prob = cvxpy.Problem(objective,constraints)
p_star_1 = prob.solve()



acc_stats = []
for _, row in df_op.iterrows():
    x_star, nu_star = row[['x_star','nu_star']]
    in_domain = all(x_star > 0) # ie. in domain of objective

    ### check kkt conditions
    ## primal feasible (equality)
    p_f_eq = np.allclose(np.dot(A,x_star),b)

    ## dual feasible (none)

    ## complementary slackness (none)

    ## stationarity
    # np.allclose(_grad(c, x_star).ravel(), np.dot(A.T,nu_star).ravel()) # too imprecise
    st_norm = np.linalg.norm(_grad(c, x_star).ravel() - np.dot(A.T,nu_star).ravel(), ord=2)

    ## cvxpy
    p_star = _f(c,x_star)
    cvx_p_star = _f(c,x.value)
    cvx_st_norm = np.linalg.norm(_grad(c, np.array(x.value)).ravel() - np.dot(A.T,np.array(constraints[0].dual_value)).ravel(), ord=2)
    entry = (in_domain, p_f_eq, st_norm, p_star, cvx_p_star, cvx_st_norm)
    acc_stats.append(entry)
df_stats = pd.DataFrame(acc_stats, columns=['dom','pf_eq','st_norm','p_star','cvx_p_star','cvx_st_norm'])
print df_stats
