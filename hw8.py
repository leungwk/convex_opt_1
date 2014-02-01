import numpy as np
from numpy.random import RandomState
import pandas as pd

from lp import run_lp_center, _plot_lp_center, _f, _grad, run_lp_strict, _plot_lp_strict, lp_solve

import cvxopt
import cvxpy

image_dir = 'img/'

## generate test data
def gen_data(m, n, seed=0):
    prng = RandomState(seed)
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

    x0 = np.abs(prng.randn(n,1)) # make positive (ie. feasible)
    b = np.dot(A,x0)
    c = prng.rand(n,1)
    return A, b, c, x0
m,n = (100,500)
A, b, c, x0 = gen_data(m,n,1)



## part 1
acc_df_lp_center_stats, df_op = run_lp_center(A, b, c, x0.copy())
image_path = image_dir +'lp_-_part_1_-_centering.png'
_plot_lp_center(acc_df_lp_center_stats, df_op, image_path)




## compare against cvxpy
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
    cvx_st_norm = np.linalg.norm(_grad(c, np.array(x.value)).ravel() - np.dot(A.T,np.array(constraints[0].dual_value)).ravel())
    entry = (in_domain, p_f_eq, st_norm, float(p_star), float(cvx_p_star), cvx_st_norm)
    acc_stats.append(entry)
df_stats = pd.DataFrame(acc_stats, columns=['dom','pf_eq','st_norm','p_star','cvx_p_star','cvx_st_norm'])
print "Part 1: Centering in a LP (comparison against cvxpy)"
print df_stats



## part 2
def orig_f(c,x): # the "f" in lp.py includes the barrier; this is the original objective
    return np.dot(c.T,x)
m,n = (100,500)
A, b, c, x0 = gen_data(m,n,1)
acc_df_lp_strict_stats, df_op_lp_strict = run_lp_strict(A, b, c, x0.copy()) # use default starting values
image_path = image_dir +'lp_-_part_2_-_lp_strict_start.png'
_plot_lp_strict(acc_df_lp_strict_stats, df_op_lp_strict, image_path)


def lp_cvx(A, b, c):
    """for comparing against cvxpy"""
    x = cvxpy.Variable(n,name='x')
    A1 = cvxopt.matrix(A)
    c1 = cvxopt.matrix(c)
    ones = cvxopt.matrix(np.ones(n))
    b1 = cvxpy.Variable(b.shape[0],name='b')
    objective = cvxpy.Minimize(
        c1.T*x
    )
    constraints = [A1*x == b1, x >= 0]
    prob = cvxpy.Problem(objective,constraints)
    p_star_1 = prob.solve()
    return x.value
x_cvx = lp_cvx(A, b, c)


acc_stats = []
for _, row in df_op_lp_strict.iterrows():
    x_star, nu_star, mu = row[['x_star','nu_star','mu']]
    ## check feasibility
    ieq_feas = all(x_star >= 0)
    eq_diff = np.linalg.norm(np.dot(A,x_star)-b)
    # np.linalg.norm(np.dot(A,df_op_lp_strict.ix[0,'x_star'])-b) # should be small
    ##
    p_star_diff = orig_f(c, x_star) -orig_f(c, x_cvx)
    row = (float(p_star_diff), mu, ieq_feas, eq_diff)
    acc_stats.append( row )
df_stats = pd.DataFrame(acc_stats, columns=['p_star_diff','mu','ieq_feas','eq_diff'])
print "Part 2: LP with strictly feasible starting point (comparison against cvxpy)"
print df_stats




# part 3
acc_stats = []
m,n = (100,500)
for seed in xrange(4):
    A, b, c, _ = gen_data(m, n, seed)
    x_star, nu_star, df = lp_solve(A, b, c)

    ieq_feas = all(x_star >= 0)
    eq_diff = np.linalg.norm(np.dot(A,x_star)-b)

    x_cvx = lp_cvx(A, b, c)

    p_star_diff = orig_f(c, x_star) -orig_f(c, x_cvx)
    row = (float(p_star_diff), True, ieq_feas, eq_diff)
    acc_stats.append( row )
## generate infeasible solution
A, b, c, _ = gen_data(m, n, 4)
A[-1,:] = np.ones(n) # append row
try:
    _ = lp_solve(A, b, c)
except ValueError as ve:
    print "lp_solve() detected infeasible system with t={0}".format(ve.args[1])
else:
    raise AssertionError("System is infeasible but lp_solve raised no exception")

df_stats = pd.DataFrame(acc_stats, columns=['p_star_diff','feas','ieq_feas','eq_diff'])
df_stats.index.name = 'seed'
print "Part 3: LP solver with a phase I method (comparison against cvxpy)"
print df_stats
